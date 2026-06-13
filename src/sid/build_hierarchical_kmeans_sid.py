"""Build near-unique hierarchical semantic IDs for tracks.

The SID format is:

    <SID0_a> <SID1_b> <SID2_c> <SID3_d>

The first `semantic_depth` codes are produced by recursively clustering inside
the parent cluster. The final leaf code is a deterministic per-leaf local item
index, which makes the SID map back to one concrete track when the leaf is not
too large.

This is a practical generative-retrieval tokenizer:
  - prefix tokens preserve coarse-to-fine music semantics;
  - final token disambiguates individual tracks;
  - the resulting SID set can be put into a trie for constrained decoding.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from build_residual_kmeans_sid import TRACK_FIELDS, load_track_matrix


def sid_tokens(codes: list[int] | np.ndarray) -> list[str]:
    return [f"<SID{i}_{int(c)}>" for i, c in enumerate(codes)]


def sid_str(codes: list[int] | np.ndarray) -> str:
    return " ".join(sid_tokens(codes))


def fit_node_kmeans(
    x: np.ndarray,
    idx: np.ndarray,
    branch: int,
    batch_size: int,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    """Return cluster labels for x[idx]."""
    n = len(idx)
    if n <= branch:
        return np.arange(n, dtype=np.int32)

    km = MiniBatchKMeans(
        n_clusters=branch,
        batch_size=min(batch_size, max(branch * 4, n)),
        max_iter=max_iter,
        n_init=1,
        random_state=seed,
        verbose=0,
    )
    return km.fit_predict(x[idx]).astype(np.int32, copy=False)


def build_codes(
    x: np.ndarray,
    track_ids: list[str],
    semantic_depth: int,
    branch: int,
    leaf_code_size: int,
    batch_size: int,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Build position-specific hierarchical SID codes.

    `semantic_depth=3` plus one leaf code gives total SID depth 4.
    """
    n = x.shape[0]
    total_depth = semantic_depth + 1
    codes = np.zeros((n, total_depth), dtype=np.int32)

    # Work queue entries: (indices, level, prefix_tuple)
    q: deque[tuple[np.ndarray, int, tuple[int, ...]]] = deque()
    q.append((np.arange(n, dtype=np.int32), 0, ()))

    leaf_buckets: list[np.ndarray] = []
    split_nodes = 0
    deterministic_nodes = 0

    while q:
        idx, level, prefix = q.popleft()
        if level == semantic_depth:
            leaf_buckets.append(idx)
            continue

        labels = fit_node_kmeans(
            x=x,
            idx=idx,
            branch=branch,
            batch_size=batch_size,
            max_iter=max_iter,
            seed=seed + split_nodes + level * 100003,
        )
        used = sorted(set(int(v) for v in labels.tolist()))
        if len(idx) <= branch:
            deterministic_nodes += 1
        else:
            split_nodes += 1

        print(
            f"[hier] level={level} prefix={prefix or 'ROOT'} "
            f"n={len(idx)} children={len(used)}",
            flush=True,
        )

        for lab in used:
            child_pos = np.where(labels == lab)[0]
            child_idx = idx[child_pos]
            codes[child_idx, level] = lab
            q.append((child_idx, level + 1, prefix + (lab,)))

    too_large = [len(b) for b in leaf_buckets if len(b) > leaf_code_size]
    if too_large:
        max_bucket = max(too_large)
        raise RuntimeError(
            f"Leaf bucket too large for final code: max={max_bucket}, "
            f"leaf_code_size={leaf_code_size}. Increase --branch, "
            f"--semantic_depth, or --leaf_code_size."
        )

    for bucket in leaf_buckets:
        # Deterministic local assignment by track_id, not by dataset order.
        ordered = sorted(bucket.tolist(), key=lambda i: track_ids[i])
        for local_code, row_idx in enumerate(ordered):
            codes[row_idx, semantic_depth] = local_code

    stats = {
        "split_nodes": split_nodes,
        "deterministic_nodes": deterministic_nodes,
        "leaf_count": len(leaf_buckets),
        "max_leaf_size": max(len(b) for b in leaf_buckets),
        "mean_leaf_size": float(np.mean([len(b) for b in leaf_buckets])),
        "leaf_size_histogram": {
            str(k): int(v)
            for k, v in sorted(Counter(len(b) for b in leaf_buckets).items())
        },
    }
    return codes, stats


def write_outputs(
    out_dir: Path,
    track_ids: list[str],
    codes: np.ndarray,
    field: str,
    args: argparse.Namespace,
    stats: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    track_to_sid: dict[str, dict] = {}
    sid_to_tracks: dict[str, list[str]] = defaultdict(list)
    for tid, row_codes in zip(track_ids, codes):
        code_list = [int(c) for c in row_codes]
        s = sid_str(code_list)
        track_to_sid[tid] = {
            "sid": code_list,
            "sid_tokens": sid_tokens(code_list),
            "sid_str": s,
        }
        sid_to_tracks[s].append(tid)

    with open(out_dir / "track_to_sid.json", "w", encoding="utf-8") as f:
        json.dump(track_to_sid, f, ensure_ascii=False, indent=2)
    with open(out_dir / "sid_to_tracks.json", "w", encoding="utf-8") as f:
        json.dump(dict(sid_to_tracks), f, ensure_ascii=False, indent=2)
    with open(out_dir / "track_ids.json", "w", encoding="utf-8") as f:
        json.dump(track_ids, f, ensure_ascii=False)
    np.save(out_dir / "codes.npy", codes)

    sid_bucket_hist = Counter(len(v) for v in sid_to_tracks.values())
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "hierarchical_kmeans_with_leaf_code",
        "embedding_field": field,
        "n_tracks": len(track_ids),
        "semantic_depth": int(args.semantic_depth),
        "total_sid_depth": int(args.semantic_depth + 1),
        "branch": int(args.branch),
        "leaf_code_size": int(args.leaf_code_size),
        "batch_size": int(args.batch_size),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "unique_sid_count": len(sid_to_tracks),
        "collision_sid_count": sum(1 for v in sid_to_tracks.values() if len(v) > 1),
        "collided_track_count": sum(len(v) for v in sid_to_tracks.values() if len(v) > 1),
        "sid_bucket_size_histogram": {str(k): int(v) for k, v in sorted(sid_bucket_hist.items())},
        "hierarchy_stats": stats,
        "outputs": {
            "track_to_sid": "track_to_sid.json",
            "sid_to_tracks": "sid_to_tracks.json",
            "track_ids": "track_ids.json",
            "codes": "codes.npy",
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[out] wrote {out_dir}")
    print(
        f"[stats] unique_sids={metadata['unique_sid_count']}/{len(track_ids)} "
        f"collisions={metadata['collision_sid_count']} "
        f"max_leaf={stats['max_leaf_size']}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="metadata-qwen3_embedding_0.6b", choices=TRACK_FIELDS)
    p.add_argument("--out_dir", default="exp/sid/hier_metadata_qwen3_sd3_b32_leaf256")
    p.add_argument("--semantic_depth", type=int, default=3)
    p.add_argument("--branch", type=int, default=32)
    p.add_argument("--leaf_code_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_iter", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    track_ids, x = load_track_matrix(args.field)
    codes, stats = build_codes(
        x=x,
        track_ids=track_ids,
        semantic_depth=args.semantic_depth,
        branch=args.branch,
        leaf_code_size=args.leaf_code_size,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    write_outputs(Path(args.out_dir), track_ids, codes, args.field, args, stats)


if __name__ == "__main__":
    main()
