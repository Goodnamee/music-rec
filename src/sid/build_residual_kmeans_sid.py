"""Build semantic item IDs for tracks with residual KMeans.

This is an RQ-style semantic tokenizer for the music catalog:

    track embedding -> [code_1, code_2, ..., code_depth]

Each level fits KMeans on the current residual. After assigning a code, the
level's centroid is subtracted from the residual, so later codes capture finer
details not explained by earlier codes.

Outputs:
  - track_to_sid.json: track_id -> SID and token string
  - sid_to_tracks.json: SID token string -> list[track_id]
  - codebooks.npz: KMeans centroids per level
  - codes.npy: integer code matrix aligned with track_ids.json
  - metadata.json: parameters and collision statistics
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


TRACK_FIELDS = [
    "audio-laion_clap",
    "image-siglip2",
    "cf-bpr",
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-9, None)


def load_track_matrix(field: str) -> tuple[list[str], np.ndarray]:
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split="all_tracks",
    )
    track_ids: list[str] = []
    vectors: list[list[float]] = []
    skipped = 0
    dim = None

    for row in tqdm(ds, desc=f"loading {field}"):
        vec = row.get(field)
        if not isinstance(vec, list) or not vec:
            skipped += 1
            continue
        if dim is None:
            dim = len(vec)
        if len(vec) != dim:
            skipped += 1
            continue
        track_ids.append(row["track_id"])
        vectors.append(vec)

    if not vectors:
        raise RuntimeError(f"No valid embeddings found for field={field}")

    x = np.asarray(vectors, dtype=np.float32)
    x = l2_normalize(x)
    if skipped:
        print(f"[warn] skipped {skipped} rows with missing/malformed vectors")
    print(f"[data] tracks={len(track_ids)} dim={x.shape[1]} field={field}")
    return track_ids, x


def fit_residual_kmeans(
    x: np.ndarray,
    depth: int,
    n_clusters: int,
    batch_size: int,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Fit residual KMeans and return codes, codebooks, reconstructed vectors."""
    residual = x.copy()
    codes = np.zeros((x.shape[0], depth), dtype=np.int32)
    codebooks: list[np.ndarray] = []
    reconstruction = np.zeros_like(x)

    for level in range(depth):
        print(
            f"[kmeans] level={level + 1}/{depth} "
            f"clusters={n_clusters} residual_norm={np.linalg.norm(residual, axis=1).mean():.4f}",
            flush=True,
        )
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init=3,
            random_state=seed + level,
            verbose=0,
        )
        level_codes = km.fit_predict(residual)
        centers = km.cluster_centers_.astype(np.float32, copy=False)
        assigned = centers[level_codes]

        codes[:, level] = level_codes.astype(np.int32, copy=False)
        codebooks.append(centers)
        reconstruction += assigned
        residual -= assigned

        mse = float(np.mean((x - reconstruction) ** 2))
        used = int(len(set(level_codes.tolist())))
        print(f"[kmeans] level={level + 1} used_clusters={used}/{n_clusters} mse={mse:.6f}")

    return codes, codebooks, reconstruction


SID_LEVEL_PREFIXES = ["a", "b", "c", "d", "e", "f", "g", "h"]  # per-level prefixes

def sid_to_token(codes: list[int] | np.ndarray, _prefix: str = None) -> str:
    """Convert integer codes to per-level SID token string: <a_7> <b_48> <c_80> <d_139>."""
    return " ".join(f"<{SID_LEVEL_PREFIXES[i]}_{int(c)}>" for i, c in enumerate(codes))


def write_outputs(
    out_dir: Path,
    track_ids: list[str],
    codes: np.ndarray,
    codebooks: list[np.ndarray],
    reconstruction: np.ndarray,
    field: str,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    track_to_sid: dict[str, dict] = {}
    sid_to_tracks: dict[str, list[str]] = defaultdict(list)
    for tid, row_codes in zip(track_ids, codes):
        sid = [int(c) for c in row_codes]
        sid_str = sid_to_token(sid)
        track_to_sid[tid] = {
            "sid": sid,
            "sid_str": sid_str,
        }
        sid_to_tracks[sid_str].append(tid)

    with open(out_dir / "track_to_sid.json", "w", encoding="utf-8") as f:
        json.dump(track_to_sid, f, ensure_ascii=False, indent=2)
    with open(out_dir / "sid_to_tracks.json", "w", encoding="utf-8") as f:
        json.dump(dict(sid_to_tracks), f, ensure_ascii=False, indent=2)
    with open(out_dir / "track_ids.json", "w", encoding="utf-8") as f:
        json.dump(track_ids, f, ensure_ascii=False)

    np.save(out_dir / "codes.npy", codes)
    np.savez(
        out_dir / "codebooks.npz",
        **{f"level_{i}": cb for i, cb in enumerate(codebooks)},
    )

    counts = Counter(len(v) for v in sid_to_tracks.values())
    n_unique = len(sid_to_tracks)
    n_tracks = len(track_ids)
    n_collided_tracks = sum(len(v) for v in sid_to_tracks.values() if len(v) > 1)
    recon_norm = np.linalg.norm(reconstruction, axis=1).mean()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "residual_kmeans",
        "embedding_field": field,
        "n_tracks": n_tracks,
        "depth": int(args.depth),
        "n_clusters": int(args.n_clusters),
        "token_prefixes": SID_LEVEL_PREFIXES[:len(codes[0])],
        "batch_size": int(args.batch_size),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "unique_sid_count": n_unique,
        "collision_sid_count": sum(1 for v in sid_to_tracks.values() if len(v) > 1),
        "collided_track_count": n_collided_tracks,
        "collision_bucket_size_histogram": {str(k): int(v) for k, v in sorted(counts.items())},
        "mean_reconstruction_norm": float(recon_norm),
        "outputs": {
            "track_to_sid": "track_to_sid.json",
            "sid_to_tracks": "sid_to_tracks.json",
            "track_ids": "track_ids.json",
            "codes": "codes.npy",
            "codebooks": "codebooks.npz",
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[out] wrote {out_dir}")
    print(
        f"[stats] unique_sids={n_unique}/{n_tracks} "
        f"collision_sids={metadata['collision_sid_count']} "
        f"collided_tracks={n_collided_tracks}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="metadata-qwen3_embedding_0.6b", choices=TRACK_FIELDS)
    p.add_argument("--out_dir", default="exp/sid/rqkmeans_metadata_qwen3_d4_k256")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n_clusters", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_iter", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    track_ids, x = load_track_matrix(args.field)
    codes, codebooks, reconstruction = fit_residual_kmeans(
        x,
        depth=args.depth,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    write_outputs(
        Path(args.out_dir),
        track_ids,
        codes,
        codebooks,
        reconstruction,
        args.field,
        args,
    )


if __name__ == "__main__":
    main()
