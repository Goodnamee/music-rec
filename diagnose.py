"""Diagnose: model prediction vs label on a training sample."""
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR = "out/sid_generator/checkpoint-500"
BASE_MODEL = "./Qwen3-0.6B"
PT_PATH = "data/sid_train_512.pt"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load training sample
data = torch.load(PT_PATH, weights_only=True)
ids = data["input_ids"][0]
labels = data["labels"][0]
prompt_len = sum(1 for l in labels if l == -100)
label_ids = labels[prompt_len:]
label_tokens = [t for t in label_ids if t != -100]
label_text = tokenizer.decode(label_tokens, skip_special_tokens=False).strip()

print(f"[data] seq_len={len(ids)}, prompt_len={prompt_len}, label_tokens={label_tokens}")
print(f"[data] label SID: {label_text}")

# Decode the prompt (first prompt_len tokens)
prompt_ids = ids[:prompt_len]
prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
print(f"[data] prompt (last 200 chars): ...{prompt_text[-200:]}")

# Load model (4-bit) — must add SID tokens first so vocab size matches
import json
with open(Path(MODEL_DIR) / "sid_config.json") as f:
    sid_config = json.load(f)
sid_tokens = sid_config["sid_tokens"]
tokenizer.add_tokens(sid_tokens)
print(f"[model] added {len(sid_tokens)} SID tokens, vocab={len(tokenizer)}")

print("[model] loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config,
    trust_remote_code=True, attn_implementation="sdpa", device_map="auto",
)
base_model.resize_token_embeddings(len(tokenizer))
print(f"[model] resized embeddings to {base_model.get_input_embeddings().weight.shape}")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)

# Load embed_weight
embed_path = Path(MODEL_DIR) / "embed_weight.pt"
if embed_path.exists():
    embed = torch.load(embed_path, map_location="cpu")
    model.get_input_embeddings().weight.data.copy_(embed.to(model.device))
    print("[model] loaded embed_weight.pt")

# Generate (unconstrained greedy)
inputs = tokenizer(prompt_text[:-4], return_tensors="pt", truncation=True, max_length=512).to(model.device)
print(f"[gen] input_len={inputs['input_ids'].shape[1]}")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=8,
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
new_tokens = output[0][inputs["input_ids"].shape[1]:]
pred_text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
print(f"[pred] raw tokens: {new_tokens.tolist()}")
print(f"[pred] decoded: {pred_text}")
print(f"[label] decoded: {label_text}")
print(f"[gen] MATCH? {pred_text.strip() == label_text.strip()}")

# ── Check rank of correct SID tokens at each position ──
print()
print("=== Logit ranking of correct SID tokens ===")
label_tokens_clean = label_tokens
with torch.no_grad():
    logits = model(**inputs).logits  # [1, seq_len, vocab]

# Check logits at the LAST input position (where first SID token should be predicted)
last_logits = logits[0, -1, :]
for i, correct_id in enumerate(label_tokens_clean):
    if correct_id == 220:  # space
        continue
    rank = (last_logits > last_logits[correct_id]).sum().item() + 1
    top5 = last_logits.topk(5).indices.tolist()
    top5_decoded = [tokenizer.decode([t]) for t in top5]
    prob = torch.softmax(last_logits.float(), -1)[correct_id].item()
    print(f"  correct={tokenizer.decode([correct_id])} (id={correct_id}), rank={rank}/{last_logits.shape[0]}, prob={prob:.6f}")
    print(f"         top-5: {top5_decoded}")

# Check lm_head weight for SID tokens
print()
print("=== LM head weight check for SID tokens ===")
lm_head = model.get_output_embeddings().weight
embed = model.get_input_embeddings().weight
for correct_id in [151673, 152064, 152243]:
    print(f"  <token {correct_id}> lm_head norm={lm_head[correct_id].norm().item():.4f}, embed norm={embed[correct_id].norm().item():.4f}")
# Compare with random text token
for text_id in [1090, 287, 30138]:
    print(f"  <text {text_id}> lm_head norm={lm_head[text_id].norm().item():.4f}, embed norm={embed[text_id].norm().item():.4f}")
