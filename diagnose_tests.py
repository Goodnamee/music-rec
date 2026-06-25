"""Tests 1 & 5: can model memorize SIDs? + do SID codes carry semantics?

Test 1: Overfit 100 samples, check if model can memorize
Test 5: Check if similar SIDs → similar tracks (RQ-VAE quality)
"""
import json
import torch
from pathlib import Path
from collections import defaultdict

# ============================================================
# TEST 5: SID semantic check — do nearby SIDs share genre/artist?
# ============================================================
print("=" * 60)
print("TEST 5: Do SID codes carry semantic meaning?")
print("=" * 60)

# Load SID mappings
with open("exp/sid/rqvae_2176d_d4_k256_3tok/sid_to_tracks.json") as f:
    sid_to_tracks = json.load(f)

# Load track metadata (try local cache first)
try:
    from datasets import load_dataset
    print("Loading track metadata...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                       split="all_tracks", streaming=True)
    track_info = {}
    for row in ds:
        tid = row["track_id"]
        genres = row.get("genres", [])
        if isinstance(genres, str):
            genres = [genres]
        artist = row.get("artist_name", "")
        track_info[tid] = {"genres": genres, "artist": artist}
        if len(track_info) % 10000 == 0:
            print(f"  loaded {len(track_info)} tracks...")
    print(f"  total: {len(track_info)} tracks with metadata")
except Exception as e:
    print(f"  Cannot load metadata: {e}")
    track_info = {}

# Analyze SID structure
# For each SID, check genre consistency of its tracks
sid_genres = {}
for sid_str, track_list in sid_to_tracks.items():
    all_genres = set()
    for tid in track_list[:5]:  # first 5 tracks per SID
        if tid in track_info:
            for g in track_info[tid]["genres"][:3]:
                all_genres.add(g)
    sid_genres[sid_str] = all_genres

# Count SIDs with consistent genres (at least 1 shared genre)
consistent = 0
total_sids = 0
for sid, genres in sid_genres.items():
    if len(sid) > 0:  # non-empty SID
        total_sids += 1
        if len(genres) >= 1:
            consistent += 1

print(f"\nSIDs with genre info: {total_sids}")
print(f"SIDs with ≥1 shared genre: {consistent} ({consistent/total_sids*100:.1f}%)")

# Check: do SIDs with same prefix (e.g., <a_7>) share genres?
prefix_genres = defaultdict(set)
for sid_str in sid_to_tracks:
    parts = sid_str.split()
    if len(parts) >= 1:
        prefix_a = parts[0]  # e.g., <a_7>
        for tid in sid_to_tracks[sid_str][:5]:
            if tid in track_info:
                for g in track_info[tid]["genres"][:3]:
                    prefix_genres[prefix_a].add(g)

print(f"\nPrefix-level genre diversity:")
for prefix in sorted(prefix_genres.keys(), key=lambda x: len(prefix_genres[x]))[:5]:
    print(f"  {prefix}: {len(prefix_genres[prefix])} unique genres")
for prefix in sorted(prefix_genres.keys(), key=lambda x: len(prefix_genres[x]))[-5:]:
    print(f"  {prefix}: {len(prefix_genres[prefix])} unique genres")

# Check: random SID vs same-prefix SID genre overlap
import random
random.seed(42)
sids = list(sid_to_tracks.keys())
same_pref_overlap = []
diff_pref_overlap = []
for _ in range(1000):
    s1 = random.choice(sids)
    s2 = random.choice(sids)
    p1, p2 = s1.split()[0] if s1 else "", s2.split()[0] if s2 else ""

    g1 = sid_genres.get(s1, set())
    g2 = sid_genres.get(s2, set())
    if g1 and g2:
        overlap = len(g1 & g2) / max(len(g1 | g2), 1)
        if p1 == p2:
            same_pref_overlap.append(overlap)
        else:
            diff_pref_overlap.append(overlap)

if same_pref_overlap and diff_pref_overlap:
    print(f"\nSame prefix genre overlap: {sum(same_pref_overlap)/len(same_pref_overlap):.3f}")
    print(f"Diff prefix genre overlap: {sum(diff_pref_overlap)/len(diff_pref_overlap):.3f}")
    print(f"Ratio: {sum(same_pref_overlap)/len(same_pref_overlap) / max(sum(diff_pref_overlap)/len(diff_pref_overlap), 0.001):.1f}x")
    if sum(same_pref_overlap)/len(same_pref_overlap) > sum(diff_pref_overlap)/len(diff_pref_overlap) * 1.5:
        print(">>> SID codes DO carry semantic meaning (same prefix → more genre overlap)")
    else:
        print(">>> SID codes carry LITTLE semantic meaning (same prefix ≈ random genre overlap)")


# ============================================================
# TEST 1: Overfitting — can model memorize 100 samples?
# ============================================================
print()
print("=" * 60)
print("TEST 1: Can model memorize 100 training samples?")
print("=" * 60)

# Load training data
data = torch.load("data/sid_train_512.pt", weights_only=True)
n_samples = len(data["input_ids"])
print(f"Training data: {n_samples} total samples, using 100 for overfit test")

# Take 100 samples (first 100)
N = 100
subset = {
    "input_ids": data["input_ids"][:N],
    "attention_mask": data["attention_mask"][:N],
    "labels": data["labels"][:N],
}
torch.save(subset, "data/sid_overfit_100.pt")
print(f"Saved 100 samples to data/sid_overfit_100.pt")

print()
print("Now run:")
print("  python src/sid/train_sid_generator.py \\")
print("    --train_pt data/sid_overfit_100.pt \\")
print("    --eval_pt data/sid_overfit_100.pt \\")
print("    --model_path ./Qwen3-0.6B \\")
print("    --output_dir out/sid_overfit \\")
print("    --preset test \\")
print("    --epochs 50 --lr 1e-3")
print()
print("Then check inference on these 100 samples — if model outputs correct SIDs,")
print("architecture is fine, problem is generalization.")
