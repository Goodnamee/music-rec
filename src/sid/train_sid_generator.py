"""Train Qwen3-0.6B + LoRA to generate SID tokens from dialogue.

Two modes:
  Local (laptop 5070 Ti): QLoRA 4-bit + SDPA + small batches
  5090:                 fp16 + Flash Attn 2 + large batches

Usage:
  # Pre-tokenize
  python src/sid/train_sid_generator.py --pre_tokenize --train_jsonl data/sid_train.jsonl

  # Local training
  python src/sid/train_sid_generator.py --train_pt data/sid_train_512.pt ...

  # 5090 training
  python src/sid/train_sid_generator.py --train_pt data/sid_train_512.pt --preset 5090 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

# 可选：liger-kernel fused CE loss，省 80% logits 显存，无损精度
_LIGER_AVAILABLE = False
try:
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy_loss as _fused_ce
    _LIGER_AVAILABLE = True
    print("[liger] fused CE kernel available")
except ImportError:
    print("[liger] not installed, fallback to standard loss")


class LigerTrainer(Trainer):
    """Custom Trainer that uses liger-kernel fused linear+CE loss.

    Instead of materializing [B*T, V] logits in fp32, it passes the last
    hidden state + lm_head.weight directly to liger's fused kernel which
    computes loss on-the-fly — saving 80%+ of the loss VRAM.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None or not _LIGER_AVAILABLE:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Forward through transformer body (skips lm_head → no logits materialized)
        outputs = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        hidden_states = outputs.last_hidden_state

        loss = _fused_ce(
            hidden_states,
            model.lm_head.weight,
            labels,
            ignore_index=-100,
        )
        return (loss, None) if return_outputs else loss

SID_CODEBOOK_SIZE = 256
SID_TOKENS = [f"<SID_{i}>" for i in range(SID_CODEBOOK_SIZE)]
# SID_DEPTH = number of SID tokens per track — auto-detected from training data output.
# e.g. "<SID_7> <SID_48> <SID_80>" → depth = 3
SID_DEPTH = None  # will be set after counting tokens in first sample

PROMPT = """Given the conversation history and user preferences, predict the semantic ID of the next recommended track.

{input}

SID:"""

PRESETS = {
    "5090": {
        "batch_size": 64, "grad_accum": 2,
        "eval_batch_size": 8, "use_4bit": False,
        "attn": "sdpa", "fp16": False, "bf16": True,
        "skip_eval": False,
    },
    "local": {
        "batch_size": 8, "grad_accum": 4,
        "eval_batch_size": 4, "use_4bit": True,
        "attn": "sdpa", "fp16": False,
        "skip_eval": False,
    },
    "test": {
        "batch_size": 8, "grad_accum": 4,
        "eval_batch_size": 4, "use_4bit": True,
        "attn": "sdpa", "fp16": False,
        "skip_eval": True,
    },
}


def tokenize_sample(input_text: str, output_text: str, tokenizer, max_length: int):
    full_text = PROMPT.format(input=input_text) + " " + output_text + tokenizer.eos_token
    tok = tokenizer(full_text, max_length=max_length, truncation=True, padding=False)
    input_ids = tok["input_ids"]

    prompt_text = PROMPT.format(input=input_text) + " "
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[:len(input_ids)] + [-100] * max(0, len(input_ids) - len(labels))

    return {"input_ids": input_ids, "attention_mask": tok["attention_mask"], "labels": labels}


def pre_tokenize(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(SID_TOKENS)

    for split, jsonl_path, pt_path in [
        ("train", args.train_jsonl, args.train_pt),
        ("eval", args.eval_jsonl, args.eval_pt),
    ]:
        ids_list, attn_list, labels_list = [], [], []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"tokenizing {split}"):
                obj = json.loads(line)
                tok = tokenize_sample(obj["input"], obj["output"], tokenizer, args.max_length)
                ids_list.append(tok["input_ids"])
                attn_list.append(tok["attention_mask"])
                labels_list.append(tok["labels"])

        torch.save({
            "input_ids": ids_list, "attention_mask": attn_list, "labels": labels_list,
        }, pt_path)
        print(f"[tokenized] {len(ids_list)} → {pt_path}")


class PTDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path: str):
        data = torch.load(pt_path, weights_only=True)
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def data_collator(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    return {
        "input_ids": torch.tensor([x["input_ids"] + [0] * (max_len - len(x["input_ids"])) for x in batch]),
        "attention_mask": torch.tensor([x["attention_mask"] + [0] * (max_len - len(x["attention_mask"])) for x in batch]),
        "labels": torch.tensor([x["labels"] + [-100] * (max_len - len(x["labels"])) for x in batch]),
    }


def compute_metrics(eval_pred):
    """Compute SID token accuracy from generated token IDs."""
    predictions, labels = eval_pred
    # predictions is list of token IDs from generate(), labels is list of label token IDs
    # Filter to SID token positions only
    correct = total = 0
    for pred_ids, label_ids in zip(predictions, labels):
        for p, l in zip(pred_ids, label_ids):
            if l != -100:
                total += 1
                if p == l:
                    correct += 1
    return {"sid_accuracy": correct / total if total > 0 else 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pre_tokenize", action="store_true")
    p.add_argument("--train_jsonl", default="data/sid_train.jsonl")
    p.add_argument("--eval_jsonl", default="data/sid_eval.jsonl")
    p.add_argument("--train_pt", default="data/sid_train_512.pt")
    p.add_argument("--eval_pt", default="data/sid_eval_512.pt")
    p.add_argument("--model_path", default="Qwen/Qwen3-0.6B",
                   help="Local path or HF model ID")
    p.add_argument("--output_dir", default="out/sid_generator")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--preset", choices=["local", "5090", "test"], default="local")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=500)
    args = p.parse_args()

    preset = PRESETS[args.preset]

    if args.pre_tokenize:
        pre_tokenize(args)
        return

    print(f"[preset] {args.preset}: batch={preset['batch_size']}×{preset['grad_accum']}, "
          f"4bit={preset['use_4bit']}, attn={preset['attn']}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(SID_TOKENS)

    # Model
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation=preset["attn"],
        device_map="auto",
    )
    if preset["use_4bit"]:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if preset.get("bf16") else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif preset.get("bf16"):
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data
    train_dataset = PTDataset(args.train_pt)
    max_eval = preset.get("max_eval_samples", args.max_eval_samples)
    eval_dataset = PTDataset(args.eval_pt)
    if max_eval > 0 and len(eval_dataset) > max_eval:
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(max_eval))
    effective_batch = preset["batch_size"] * preset["grad_accum"]
    total_steps = (len(train_dataset) // effective_batch) * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    print(f"[data] train={len(train_dataset)}, eval={len(eval_dataset)}, steps≈{total_steps}")

    # Auto-detect SID_DEPTH: count how many <SID_*> tokens are in first sample's labels.
    # Labels are like: [-100, ..., -100, <SID_237>, " ", <SID_142>, " ", <SID_87>, <eos>]
    # So non-masked count = depth + (depth-1 spaces) + 1 eos. We count only <SID_*>.
    first_labels = train_dataset[0]["labels"]
    sid_ids = set(tokenizer.convert_tokens_to_ids(SID_TOKENS))
    sid_depth = sum(1 for t in first_labels if t in sid_ids)
    global SID_DEPTH
    SID_DEPTH = sid_depth
    print(f"[sid] auto-detected depth={SID_DEPTH} (non_masked={sum(1 for t in first_labels if t != -100)})")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=preset["batch_size"],
        per_device_eval_batch_size=preset["eval_batch_size"],
        gradient_accumulation_steps=preset["grad_accum"],
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=args.logging_steps,
        eval_strategy="no" if preset.get("skip_eval") else "steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=preset.get("fp16", False),
        bf16=preset.get("bf16", False),
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer_cls = LigerTrainer if _LIGER_AVAILABLE else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    model.config.use_cache = False

    trainer.train()

    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(Path(args.output_dir) / "sid_config.json", "w") as f:
        json.dump({"depth": SID_DEPTH, "codebook_size": SID_CODEBOOK_SIZE, "sid_tokens": SID_TOKENS}, f)

    if not preset.get("skip_eval") and args.max_steps < 0:
        result = trainer.evaluate()
        print(f"[final] eval_sid_accuracy = {result.get('eval_sid_accuracy', 'N/A')}")


if __name__ == "__main__":
    main()
