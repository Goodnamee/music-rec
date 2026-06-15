"""Prepare 3-modal track embedding for RQ-VAE training.

Loads attributes-qwen3 (1024d), lyrics-qwen3 (1024d), cf-bpr (128d)
from HuggingFace in a single pass, L2-normalizes each, concats to 2176d,
saves as .npy + track_ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

FIELDS = [
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "cf-bpr",
]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-9, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="exp/sid/multimodal_2176d")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split="all_tracks",
    )

    track_ids = []
    vectors = {f: [] for f in FIELDS}
    skipped = {f: 0 for f in FIELDS}

    for row in tqdm(ds, desc="loading 3 modalities", total=47071):
        valid = True
        for f in FIELDS:
            v = row.get(f)
            if not isinstance(v, list) or not v:
                skipped[f] += 1
                valid = False
        if not valid:
            continue
        track_ids.append(row["track_id"])
        for f in FIELDS:
            vectors[f].append(row[f])

    # L2 normalize each modality and concat
    parts = []
    for f in FIELDS:
        x = np.asarray(vectors[f], dtype=np.float32)
        x = l2_normalize(x)
        print(f"[data] {f}: {x.shape[0]} tracks, dim={x.shape[1]}, skipped={skipped[f]}")
        parts.append(x)

    combined = np.concatenate(parts, axis=1)
    print(f"[data] combined: {combined.shape[0]} tracks, {combined.shape[1]} dim")

    np.save(out_dir / "embeddings.npy", combined)
    with open(out_dir / "track_ids.txt", "w") as f:
        for tid in track_ids:
            f.write(tid + "\n")
    print(f"[out] saved to {out_dir}")


if __name__ == "__main__":
    main()
