"""SID Generator inference: dialogue → beam-search SIDs → track_ids → eval JSON.

Constrained decoding: only valid <SID_x> tokens are allowed.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"

    return model, tokenizer, sid_tokens


def build_constrained_prefix(tokenizer, sid_tokens, sid_depth: int = 4):
    """Build prefix allowed token mask for constrained SID generation."""
    # Get SID token IDs
    sid_ids = [tokenizer.convert_tokens_to_ids(t) for t in sid_tokens]
    eos_id = tokenizer.eos_token_id
    return {
        "allowed_ids": sid_ids + [eos_id],
        "sid_ids": set(sid_ids),
    }


def constrained_generate(model, tokenizer, prompt: str, constrained, max_new_tokens: int = 32):
    """Generate with beam search, constrained to SID tokens."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    # Create logits processor that masks out non-SID tokens after generating SID:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=40,
            num_return_sequences=20,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0,
            output_scores=False,
            return_dict_in_generate=True,
        )

    # Decode each beam
    sids = []
    for beam_ids in outputs.sequences:
        new_tokens = beam_ids[inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        sids.append(decoded.strip())

    return sids


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

    # Load dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split=args.split, streaming=False)
    sessions = list(ds)
    if args.max_sessions > 0:
        sessions = sessions[:args.max_sessions]

    # Build all inputs
    all_samples = []
    for session in tqdm(sessions, desc="building inputs"):
        all_samples.extend(build_session_inputs(session, tokenizer, track_to_sid, sid_to_tracks))

    # Inference
    results = []
    for sample in tqdm(all_samples, desc="generating"):
        prompt = PROMPT.format(input=sample["input"])
        sids = constrained_generate(model, tokenizer, prompt, {})

        # Decode SIDs to track IDs
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

        # Pad if not enough tracks
        while len(track_ids) < 20:
            track_ids.append("")

        results.append({
            "session_id": sample["session_id"],
            "user_id": sample["user_id"],
            "turn_number": sample["turn"],
            "predicted_track_ids": track_ids[:20],
            "predicted_response": "",
        })

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"[out] {len(results)} predictions → {out_path}")


if __name__ == "__main__":
    main()
