"""Train RQ-VAE with MLP bottleneck + EMA codebook to build Semantic IDs.

Architecture:
    2176d → MLP encoder → 64d latent → RQ (4×256, EMA) → MLP decoder → 2176d

EMA is the key anti-collapse mechanism — gradient-based VQ collapsed to 1 code,
EMA keeps unused codes alive by maintaining a moving average per centroid.

Usage:
    python src/sid/build_rqvae_sid.py \
        --embedding exp/sid/multimodal_2176d/embeddings.npy \
        --track_ids exp/sid/multimodal_2176d/track_ids.txt \
        --out_dir exp/sid/rqvae_2176d_d4_k256
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# MLP encoder/decoder
# ---------------------------------------------------------------------------

class MLPLayers(nn.Module):
    """Stack of Linear → ReLU (no activation on last layer)."""

    def __init__(self, layer_dims: list[int], dropout: float = 0.0):
        super().__init__()
        modules = []
        for i, (in_d, out_d) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            modules.append(nn.Linear(in_d, out_d))
            if i < len(layer_dims) - 2:
                modules.append(nn.ReLU())
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# Vector Quantizer with EMA (proven anti-collapse from v2)
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """VQ with EMA codebook updates and KMeans init."""

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.eps = eps

        embedding = torch.randn(codebook_size, dim) * 0.01
        embedding = F.normalize(embedding, dim=1)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(codebook_size))
        self.register_buffer("ema_embedding", embedding.clone())

    @torch.no_grad()
    def init_kmeans(self, data: torch.Tensor):
        x = data.detach().cpu().numpy()
        print(f"[vq] kmeans init: {x.shape[0]} samples → {self.codebook_size} centroids in {self.dim}d")
        km = MiniBatchKMeans(
            n_clusters=self.codebook_size,
            batch_size=4096,
            max_iter=50,
            n_init=1,
            random_state=42,
        )
        km.fit(x)
        centers = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(data.device)
        self.embedding = centers
        self.ema_embedding = centers.clone()
        self.ema_count = torch.ones(self.codebook_size, device=data.device) * (x.shape[0] / self.codebook_size)

    def forward(
        self, z: torch.Tensor, reset_dead: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = z.shape[0]
        z_f = z.reshape(B, self.dim)

        # Distance to codebook
        dist = (
            z_f.pow(2).sum(1, keepdim=True)
            - 2 * z_f @ self.embedding.T
            + self.embedding.pow(2).sum(1)
        )
        codes = dist.argmin(dim=1)
        z_q = self.embedding[codes]

        # Commitment loss (clone to prevent corruption from in-place EMA buffer updates)
        commit_loss = (self.commitment_weight * F.mse_loss(z_f, z_q.detach(), reduction='mean')).clone()

        # Straight-through
        z_q = z + (z_q - z).detach()

        # EMA update
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(codes, self.codebook_size).float()
                count = onehot.sum(0)

                # Reset dead codes periodically
                if reset_dead and z_f.shape[0] > self.codebook_size:
                    used = set(codes.tolist())
                    dead = [i for i in range(self.codebook_size) if i not in used]
                    if dead:
                        idx = torch.randint(0, z_f.shape[0], (len(dead),), device=z_f.device)
                        self.embedding[dead] = z_f[idx] + torch.randn_like(z_f[idx]) * 0.01
                        self.ema_embedding[dead] = self.embedding[dead]
                        count[dead] += 2.0

                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * count
                n = self.ema_count.sum()
                count_stable = (self.ema_count + self.eps) / (n + self.codebook_size * self.eps) * n
                embed_sum = onehot.T @ z_f
                self.ema_embedding = self.decay * self.ema_embedding + (1 - self.decay) * embed_sum
                self.embedding = self.ema_embedding / count_stable.unsqueeze(1)

        return z_q, codes, commit_loss


# ---------------------------------------------------------------------------
# Residual Quantizer
# ---------------------------------------------------------------------------

class ResidualQuantizer(nn.Module):
    """Stack of EMA VectorQuantizers on successive residuals."""

    def __init__(
        self,
        depth: int,
        codebook_size: int,
        dim: int,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([
            VectorQuantizer(codebook_size, dim, decay, commitment_weight)
            for _ in range(depth)
        ])

    @torch.no_grad()
    def init_kmeans(self, data: torch.Tensor):
        """Init each layer with KMeans on residuals of encoded data."""
        x = data
        for i, vq in enumerate(self.layers):
            vq.init_kmeans(x)
            # Compute codes and subtract to get next residual (use CPU embedding)
            emb = vq.embedding.to(x.device)
            d = (x.pow(2).sum(1, keepdim=True)
                 - 2 * x @ emb.T
                 + emb.pow(2).sum(1))
            codes = d.argmin(dim=1)
            x = x - emb[codes]
            print(f"[rq] layer {i} residual_norm after init: {torch.norm(x, dim=1).mean():.4f}")

    def forward(
        self, z: torch.Tensor, reset_dead: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        z_q_sum = 0
        all_codes = []
        total_loss = 0.0

        for vq in self.layers:
            z_q, codes, loss = vq(residual, reset_dead=reset_dead)
            residual = residual - z_q
            z_q_sum = z_q_sum + z_q
            all_codes.append(codes)
            total_loss = total_loss + loss.clone()

        return z_q_sum, total_loss, torch.stack(all_codes, dim=1)


# ---------------------------------------------------------------------------
# RQ-VAE
# ---------------------------------------------------------------------------

class RQVAE(nn.Module):
    """RQ-VAE: MLP encoder → bottleneck → EMA RQ → MLP decoder."""

    def __init__(
        self,
        input_dim: int = 2176,
        encoder_dims: list[int] | None = None,
        latent_dim: int = 64,
        depth: int = 4,
        codebook_size: int = 256,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if encoder_dims is None:
            encoder_dims = [input_dim, 2048, 1024, 512, 256, 128, latent_dim]
        decoder_dims = encoder_dims[::-1]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.depth = depth
        self.codebook_size = codebook_size

        self.encoder = MLPLayers(encoder_dims, dropout=dropout)
        self.rq = ResidualQuantizer(depth, codebook_size, latent_dim, decay, commitment_weight)
        self.decoder = MLPLayers(decoder_dims, dropout=dropout)

    def forward(self, x: torch.Tensor, reset_dead: bool = False) -> dict:
        z = self.encoder(x)
        z_q, quant_loss, codes = self.rq(z, reset_dead=reset_dead)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        return {
            "loss": recon_loss + quant_loss,
            "recon_loss": recon_loss,
            "quant_loss": quant_loss,
            "codes": codes,
            "x_recon": x_recon,
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, _, codes = self.rq(z)
        return codes

    @torch.no_grad()
    def init_codebooks(self, data: torch.Tensor, sample_size: int = 20000):
        """Pre-init all VQ codebooks via KMeans on encoder output."""
        n = min(sample_size, data.shape[0])
        encoded = self.encoder(data[:n].to(data.device))
        self.rq.init_kmeans(encoded)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rqvae(
    model: RQVAE,
    data: torch.Tensor,
    epochs: int = 500,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: str = "cuda",
    log_every: int = 25,
    reset_interval: int = 10,
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = data.shape[0]
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = total_recon = total_quant = 0.0
        n_batches = 0
        do_reset = (epoch % reset_interval == 0)

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = data[idx].to(device)
            out = model(batch, reset_dead=do_reset)
            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += out["loss"].item()
            total_recon += out["recon_loss"].item()
            total_quant += out["quant_loss"].item()
            n_batches += 1

        scheduler.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                codes = model.encode(data.to(device)).cpu().numpy()
                layer_utils = [len(set(codes[:, l].tolist())) for l in range(model.depth)]
                total_unique = len(set(tuple(c) for c in codes))

            print(
                f"[epoch {epoch:4d}] recon={total_recon/n_batches:.6f} "
                f"quant={total_quant/n_batches:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"layer_used={layer_utils} "
                f"unique_sids={total_unique}/{n}"
            )

        history.append({
            "epoch": epoch,
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "quant_loss": total_quant / n_batches,
            "layer_utilization": layer_utils,
            "unique_sids": total_unique,
        })

    return history


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def sid_token_string(codes: list[int], prefix: str = "SID") -> str:
    return " ".join(f"<{prefix}_{int(c)}>" for c in codes)


def save_outputs(
    out_dir: Path,
    track_ids: list[str],
    codes: np.ndarray,
    model: RQVAE,
    history: list[dict],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    track_to_sid: dict = {}
    sid_to_tracks: dict = defaultdict(list)
    for tid, row_codes in zip(track_ids, codes):
        sid = [int(c) for c in row_codes]
        sid_str = sid_token_string(sid, args.token_prefix)
        track_to_sid[tid] = {"sid": sid, "sid_str": sid_str}
        sid_to_tracks[sid_str].append(tid)

    counts = Counter(len(v) for v in sid_to_tracks.values())
    n_unique = len(sid_to_tracks)
    n_tracks = len(track_ids)
    n_collided = sum(len(v) for v in sid_to_tracks.values() if len(v) > 1)

    with open(out_dir / "track_to_sid.json", "w", encoding="utf-8") as f:
        json.dump(track_to_sid, f, ensure_ascii=False, indent=1)
    with open(out_dir / "sid_to_tracks.json", "w", encoding="utf-8") as f:
        json.dump(dict(sid_to_tracks), f, ensure_ascii=False, indent=1)
    with open(out_dir / "track_ids.json", "w", encoding="utf-8") as f:
        json.dump(track_ids, f, ensure_ascii=False)
    np.save(out_dir / "codes.npy", codes)

    codebooks = {}
    for l, vq in enumerate(model.rq.layers):
        codebooks[f"level_{l}"] = vq.embedding.detach().cpu().numpy()
    np.savez(out_dir / "codebooks.npz", **codebooks)

    torch.save(model.state_dict(), out_dir / "rqvae_model.pt")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "rqvae_mlp_bottleneck_ema",
        "modalities": ["attributes-qwen3_embedding_0.6b", "lyrics-qwen3_embedding_0.6b", "cf-bpr"],
        "input_dim": int(model.input_dim),
        "latent_dim": int(model.latent_dim),
        "n_tracks": n_tracks,
        "depth": int(args.depth),
        "codebook_size": int(args.codebook_size),
        "token_prefix": args.token_prefix,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "unique_sid_count": n_unique,
        "collision_sid_count": sum(1 for v in sid_to_tracks.values() if len(v) > 1),
        "collided_track_count": n_collided,
        "collision_bucket_size_histogram": {str(k): int(v) for k, v in sorted(counts.items())},
        "final_recon_loss": history[-1]["recon_loss"],
        "history": history,
        "outputs": {
            "track_to_sid": "track_to_sid.json",
            "sid_to_tracks": "sid_to_tracks.json",
            "track_ids": "track_ids.json",
            "codes": "codes.npy",
            "codebooks": "codebooks.npz",
            "model": "rqvae_model.pt",
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[out] {out_dir}")
    print(f"[stats] unique_sids={n_unique}/{n_tracks} ({n_unique/n_tracks*100:.1f}%)")
    print(f"[stats] collision_sids={metadata['collision_sid_count']} collided_tracks={n_collided}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding", required=True)
    p.add_argument("--track_ids", required=True)
    p.add_argument("--out_dir", default="exp/sid/rqvae_2176d_d4_k256")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decay", type=float, default=0.9)
    p.add_argument("--commitment_weight", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--kmeans_sample", type=int, default=20000)
    p.add_argument("--token_prefix", default="SID")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[device] using {device}")

    # Load
    print(f"[data] loading {args.embedding}")
    data = torch.from_numpy(np.load(args.embedding))
    with open(args.track_ids) as f:
        track_ids = [line.strip() for line in f]
    print(f"[data] {data.shape[0]} tracks, dim={data.shape[1]}")
    assert data.shape[0] == len(track_ids)

    # Model
    model = RQVAE(
        input_dim=data.shape[1],
        latent_dim=args.latent_dim,
        depth=args.depth,
        codebook_size=args.codebook_size,
        decay=args.decay,
        commitment_weight=args.commitment_weight,
        dropout=args.dropout,
    )
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} params")
    print(f"[model] {data.shape[1]} → ... → {args.latent_dim}d, "
          f"{args.depth} layers × {args.codebook_size} codes")

    # KMeans pre-init
    print("[init] kmeans pre-initialization of all codebooks...")
    model.to(device)
    model.init_codebooks(data.to(device), sample_size=args.kmeans_sample)

    # Train
    history = train_rqvae(
        model, data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # Encode
    model.eval()
    with torch.no_grad():
        codes = model.encode(data.to(device)).cpu().numpy()

    save_outputs(Path(args.out_dir), track_ids, codes, model, history, args)


if __name__ == "__main__":
    main()
