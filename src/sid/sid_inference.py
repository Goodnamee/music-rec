"""SID Generator inference: dialogue → beam-search SIDs → track_ids → eval JSON.

Constrained decoding: only valid per-level SID tokens (e.g. <a_x>, <b_x>, ...) are allowed.
Beam search produces 20 diverse SID candidates naturally.

Usage:
    python src/sid/sid_inference.py \
        --model_dir out/sid_generator \
        --sid_to_tracks exp/sid/rqvae_2176d_d4_k256/sid_to_tracks.json \
        --out exp/inference/devset/sid_generator.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    LogitsProcessorList, LogitsProcessor,
)
from tqdm import tqdm

PROMPT = """Given the conversation history and user preferences, predict the semantic ID of the next recommended track.

{input}

SID:"""


def load_model(model_dir: str, base_model: str = "Qwen/Qwen3-0.6B", device: str = "cuda"):
    """Load trained SID Generator with LoRA adapter."""
    with open(Path(model_dir) / "sid_config.json") as f:
        sid_config = json.load(f)
    sid_tokens = sid_config["sid_tokens"]

    # Load base model (4-bit for memory)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # Load tokenizer first (needed for vocab size)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    # Resize to match trained tokenizer (which includes SID tokens)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, model_dir)
    # Load trained embedding weights (saved by train_sid_generator.py)
    embed_path = Path(model_dir) / "embed_weight.pt"
    if embed_path.exists():
        embed_weight = torch.load(embed_path, map_location="cpu")
        model.get_input_embeddings().weight.data.copy_(embed_weight.to(model.device))
        print(f"[model] loaded trained embedding from {embed_path}")
    model.eval()

    return model, tokenizer, sid_tokens


def build_prefix_tree(sid_to_tracks: dict, tokenizer) -> dict:
    """Build hash-based prefix tree: prefix → list of valid next token IDs.

    For SID "<a_7> <b_48> <c_80> <d_139>" (4 SID tokens):
    "<a_7>" → [space_id]
    "<a_7> " → [<b_48>]
    ...
    "<a_7> <b_48> <c_80> <d_139>" → [eos]
    """
    tree: dict[str, set] = {}
    eos_id = tokenizer.eos_token_id
    space_str = " "

    for sid_str in sid_to_tracks:
        # Tokenize the full SID string: "<a_7> <b_48> <c_80> <d_139>"
        tokens = tokenizer.encode(sid_str, add_special_tokens=False)
        # Build prefixes at each step
        for i in range(len(tokens)):
            prefix_tokens = tokens[:i]
            prefix_key = tokenizer.decode(prefix_tokens, skip_special_tokens=False)
            next_token = tokens[i]
            if prefix_key not in tree:
                tree[prefix_key] = set()
            tree[prefix_key].add(next_token)
        # Complete SID → eos
        full_key = sid_str
        if full_key not in tree:
            tree[full_key] = set()
        tree[full_key].add(eos_id)

    # Convert sets to lists
    return {k: list(v) for k, v in tree.items()}


class ConstrainedSIDLogitsProcessor(LogitsProcessor):
    """MiniOneRec-style: hash prefix tree + log_softmax + -inf mask."""
    def __init__(self, prefix_tree: dict, num_beams: int, eos_token_id: int):
        self.prefix_tree = prefix_tree
        self._num_beams = num_beams
        self.eos_token_id = eos_token_id
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))

        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                # Extract the generated prefix (last `step` tokens)
                if self.step == 0:
                    # First step: empty prefix → start of SID
                    prefix_key = ""
                else:
                    prefix_tokens = sent[-self.step:].tolist()
                    prefix_key = tokenizer_internal.decode(prefix_tokens, skip_special_tokens=False)

                allowed = self.prefix_tree.get(prefix_key, [])
                if not allowed:
                    # No valid continuation — force EOS
                    mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                else:
                    mask[batch_id * self._num_beams + beam_id, allowed] = 0

        self.step += 1
        return scores + mask


# Global tokenizer reference for use inside processor (avoid closure issues)
tokenizer_internal = None


def batch_generate(model, tokenizer, sid_tokens: list[str], prompts: list[str],
                   prefix_tree: dict, max_new_tokens: int = 8,
                   num_beams: int = 20, num_return: int = 20):
    """Generate SID sequences with hash-based constrained beam search."""
    global tokenizer_internal
    tokenizer_internal = tokenizer

    inputs = tokenizer(prompts, return_tensors="pt", truncation=True,
                       max_length=512, padding=True).to(model.device)
    B = inputs["input_ids"].shape[0]

    clp = ConstrainedSIDLogitsProcessor(prefix_tree, num_beams, tokenizer.eos_token_id)
    logits_processor = LogitsProcessorList([clp])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            output_scores=False,
            return_dict_in_generate=True,
            logits_processor=logits_processor,
        )

    # outputs.sequences shape: (B * num_return, padded_input_len + generated_len)
    # All inputs are left-padded to same max length → extract from max_len onward
    input_len = inputs["input_ids"].shape[1]
    all_sids = []
    for i in range(B):
        sample_sids = []
        for j in range(num_return):
            beam_idx = i * num_return + j
            new_tokens = outputs.sequences[beam_idx][input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if decoded:
                sample_sids.append(decoded)
        all_sids.append(sample_sids)

    return all_sids


def build_session_inputs(session: dict, tokenizer, track_to_sid: dict, sid_to_tracks: dict):
    """Build turn-by-turn inputs for a session, including previous SID context."""
    conversations = session["conversations"]
    uid = session["user_id"]

    turn_user_texts: dict[int, list[str]] = {}
    turn_music_track: dict[int, str] = {}
    for msg in conversations:
        t = msg["turn_number"]
        if t not in turn_user_texts:
            turn_user_texts[t] = []
        if msg["role"] == "user":
            turn_user_texts[t].append(msg["content"])
        elif msg["role"] == "music":
            turn_music_track[t] = msg["content"]

    # Get user culture
    user_culture = ""
    if "user_profile" in session and session["user_profile"]:
        user_culture = session["user_profile"].get("preferred_musical_culture", "")

    prev_rec_sids = []
    all_user_texts = []
    samples = []

    for turn in range(1, 9):
        if turn not in turn_user_texts:
            break
        user_text = " ".join(turn_user_texts[turn])
        all_user_texts.append(user_text)

        parts = []
        if user_culture:
            parts.append(f"User musical preference: {user_culture}")

        for i, (prev_text, prev_sid) in enumerate(zip(all_user_texts[:-1], prev_rec_sids)):
            parts.append(f"Previously recommended: {prev_sid}")
            parts.append(f"User said: {prev_text}")

        parts.append(f"User said: {all_user_texts[-1]}")

        input_text = "\n".join(parts)

        # Get current turn's music track SID for prev_rec
        track_id = turn_music_track.get(turn, "")
        sid = ""
        if track_id and track_id in track_to_sid:
            sid = track_to_sid[track_id]["sid_str"]

        samples.append({
            "session_id": session["session_id"],
            "user_id": uid,
            "turn": turn,
            "input": input_text,
            "gt_track_id": track_id,
        })

        prev_rec_sids.append(sid)

    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_path", default="Qwen/Qwen3-0.6B",
                   help="Local path or HF model ID for base model")
    p.add_argument("--sid_to_tracks", required=True)
    p.add_argument("--track_to_sid", required=True)
    p.add_argument("--out", default="exp/inference/devset/sid_generator.json")
    p.add_argument("--split", default="test")
    p.add_argument("--max_sessions", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Load model
    print("[model] loading SID Generator...")
    model, tokenizer, sid_tokens = load_model(args.model_dir, args.model_path, args.device)

    # Load mappings
    with open(args.sid_to_tracks) as f:
        sid_to_tracks = json.load(f)
    with open(args.track_to_sid) as f:
        track_to_sid = json.load(f)
    print(f"[data] {len(sid_to_tracks)} unique SIDs, {len(track_to_sid)} tracks")

    # Build prefix tree for constrained decoding
    prefix_tree = build_prefix_tree(sid_to_tracks, tokenizer)
    print(f"[tree] {len(prefix_tree)} prefix entries")

    # Load dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split=args.split, streaming=False)
    sessions = list(ds)
    if args.max_sessions > 0:
        sessions = sessions[:args.max_sessions]

    # Build all inputs
    all_samples = []
    for session in tqdm(sessions, desc="building inputs"):
        all_samples.extend(build_session_inputs(session, tokenizer, track_to_sid, sid_to_tracks))

    # Inference (batched) with incremental save — resume safe after sleep/crash
    INFER_BATCH = 1
    SAVE_EVERY = 50  # save every N items
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing partial results
    results = []
    start_idx = 0
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"[resume] {start_idx} predictions already saved, skipping")

    prompts = [PROMPT.format(input=s["input"]) for s in all_samples]

    for i in tqdm(range(start_idx, len(prompts), INFER_BATCH), desc="generating",
                  initial=start_idx, total=len(prompts)):
        batch_prompts = prompts[i : i + INFER_BATCH]
        batch_samples = all_samples[i : i + INFER_BATCH]
        batch_sids = batch_generate(model, tokenizer, sid_tokens, batch_prompts, prefix_tree)

        for sample, sids in zip(batch_samples, batch_sids):
            track_ids = []
            seen = set()
            for sid_str in sids:
                if sid_str in sid_to_tracks:
                    for tid in sid_to_tracks[sid_str]:
                        if tid not in seen:
                            track_ids.append(tid)
                            seen.add(tid)
                            if len(track_ids) >= 20:
                                break
                if len(track_ids) >= 20:
                    break
            while len(track_ids) < 20:
                track_ids.append("")

            results.append({
                "session_id": sample["session_id"],
                "user_id": sample["user_id"],
                "turn_number": sample["turn"],
                "predicted_track_ids": track_ids[:20],
                "predicted_response": "",
            })

        # Incremental save
        if (i + 1) % SAVE_EVERY == 0 or i + INFER_BATCH >= len(prompts):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=1)

    print(f"[out] {len(results)} predictions → {out_path}")


if __name__ == "__main__":
    main()
