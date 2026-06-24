"""Build SID Generator training data with context and auxiliary task.

Main task samples (per turn):
  INPUT:  user_profile + previous SID recs + user dialogue
  LABEL:  SID of ground truth track for this turn

Auxiliary task samples (per track):
  INPUT:  track attributes text
  LABEL:  SID of that track

Usage:
    python src/sid/build_training_data.py \
        --track_to_sid exp/sid/rqvae_2176d_d4_k256/track_to_sid.json \
        --out data/sid_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

TURN_SEP = "\n"

PROMPT_USER_PREF = "User musical preference: {culture}"
PROMPT_PREV_REC = "Previously recommended: {sid}"
PROMPT_USER_MSG = "User said: {text}"
PROMPT_AUX = "Track description: {desc}"


def load_sid_map(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_user_profiles() -> dict:
    """Load user_id -> preferred_musical_culture."""
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        split="train",
        streaming=True,
    )
    profiles = {}
    for row in tqdm(ds, desc="loading user profiles"):
        uid = row["user_id"]
        if uid not in profiles:
            profiles[uid] = row["user_profile"].get("preferred_musical_culture", "")
    return profiles


def build_main_samples(split: str, track_to_sid: dict, user_profiles: dict):
    """Build dialogue → SID training samples."""
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        split=split,
        streaming=False,
    )
    samples = []
    skipped = 0

    for session in tqdm(ds, desc=f"main {split}"):
        uid = session["user_id"]
        user_culture = user_profiles.get(uid, "")

        conversations = session["conversations"]

        # Collect per-turn data
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

        # Build cumulative samples
        prev_rec_sids = []  # SID strings of previously recommended tracks
        all_user_texts = []

        for turn in range(1, 9):
            if turn not in turn_user_texts or turn not in turn_music_track:
                break

            track_id = turn_music_track[turn]
            sid_entry = track_to_sid.get(track_id)
            if sid_entry is None:
                skipped += 1
                continue  # Skip turn entirely: no sample generated, no context for later turns

            user_text = " ".join(turn_user_texts[turn])
            all_user_texts.append(user_text)

            # Build input: user_pref + previous_recs + dialogue history
            parts = []
            if user_culture:
                parts.append(PROMPT_USER_PREF.format(culture=user_culture))

            for i, (prev_text, prev_sid) in enumerate(zip(all_user_texts[:-1], prev_rec_sids)):
                parts.append(PROMPT_PREV_REC.format(sid=prev_sid))
                parts.append(PROMPT_USER_MSG.format(text=prev_text))

            if all_user_texts:  # current user text
                parts.append(PROMPT_USER_MSG.format(text=all_user_texts[-1]))

            input_text = TURN_SEP.join(parts)
            output_text = sid_entry["sid_str"]

            samples.append({
                "type": "main",
                "session_id": session["session_id"],
                "turn": turn,
                "input": input_text,
                "output": output_text,
            })

            # Update for next turn
            prev_rec_sids.append(sid_entry["sid_str"])

    print(f"[main] {split}: {len(samples)} samples, skipped={skipped}")
    return samples


def build_aux_samples(track_to_sid: dict, max_samples: int = 40000):
    """Build track description → SID auxiliary samples (反向词典)."""
    # Load track metadata for descriptions
    try:
        ds = load_dataset(
            "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
            split="all_tracks",
            streaming=True,
        )
        it = iter(ds)
        row = next(it)
        print(f"[aux] metadata fields: {list(row.keys())}")
    except Exception:
        print("[aux] no metadata dataset, skipping auxiliary samples")
        return []

    samples = []
    for row in tqdm(ds, desc="aux samples", total=max_samples):
        track_id = row.get("track_id", "")
        sid_entry = track_to_sid.get(track_id)
        if sid_entry is None:
            continue

        # Build description from metadata
        parts = []
        if row.get("track_name"):
            parts.append(f"Title: {row['track_name']}")
        if row.get("artist_name"):
            parts.append(f"Artist: {row['artist_name']}")
        if row.get("album_name"):
            parts.append(f"Album: {row['album_name']}")
        if row.get("genres"):
            genres = row["genres"]
            if isinstance(genres, list):
                parts.append(f"Genres: {', '.join(genres[:5])}")
            elif isinstance(genres, str):
                parts.append(f"Genres: {genres}")

        if not parts:
            continue

        desc = ". ".join(parts)
        samples.append({
            "type": "aux",
            "session_id": "",
            "turn": 0,
            "input": PROMPT_AUX.format(desc=desc),
            "output": sid_entry["sid_str"],
        })

        if len(samples) >= max_samples:
            break

    print(f"[aux] {len(samples)} samples")
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track_to_sid", required=True)
    p.add_argument("--out", default="data/sid_train.jsonl")
    p.add_argument("--out_eval", default="data/sid_eval.jsonl")
    p.add_argument("--aux_samples", type=int, default=40000)
    args = p.parse_args()

    track_to_sid = load_sid_map(args.track_to_sid)
    print(f"[sid] {len(track_to_sid)} tracks")

    user_profiles = load_user_profiles()
    print(f"[users] {len(user_profiles)} profiles loaded")

    # Main task
    train_main = build_main_samples("train", track_to_sid, user_profiles)
    eval_main = build_main_samples("test", track_to_sid, user_profiles)

    # Auxiliary task
    aux_samples = build_aux_samples(track_to_sid, max_samples=args.aux_samples)

    # Combine
    train_all = train_main + aux_samples

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in train_all:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[out] {len(train_all)} total ({len(train_main)} main + {len(aux_samples)} aux) → {out_path}")

    eval_path = Path(args.out_eval)
    with open(eval_path, "w", encoding="utf-8") as f:
        for s in eval_main:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[out] {len(eval_main)} eval → {eval_path}")


if __name__ == "__main__":
    main()
