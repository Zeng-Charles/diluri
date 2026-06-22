#!/usr/bin/env python
"""Train a Diffusion Policy from a replay buffer of demonstrations.

Usage:
    python scripts/train.py \\
        --config  configs/policy/diffusion.yaml \\
        --buffer  data/demos.pkl \\
        --output  checkpoints/diffusion_policy.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diluri_sim.config import load_diffusion_policy_config
from diluri_sim.dataset import ReplayBuffer
from diluri_sim.policy import DiffusionPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Diffusion Policy.")
    p.add_argument("--config",  default=str(ROOT / "configs/policy/diffusion.yaml"))
    p.add_argument("--buffer",  required=True, help="Path to replay buffer .pkl")
    p.add_argument("--output",  required=True, help="Output checkpoint .pt path")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=50, help="Log loss every N epochs")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    config = load_diffusion_policy_config(args.config)

    print(f"Device : {device}")
    print(f"Config : obs_dim={config.obs_dim}  action_dim={config.action_dim}")
    print(f"         T_obs={config.T_obs}  T_pred={config.T_pred}  T_exec={config.T_exec}")
    print(f"         n_diffusion_steps={config.n_diffusion_steps}  epochs={config.n_epochs}")

    buffer = ReplayBuffer.load(args.buffer)
    print(f"Buffer : {buffer}")

    # Compute and store normalization stats
    stats = buffer.compute_stats()
    policy = DiffusionPolicy(config)
    policy.set_normalization_stats(
        stats["obs_mean"], stats["obs_std"],
        stats["action_mean"], stats["action_std"],
    )
    policy.network.to(device)

    optimizer = optim.AdamW(policy.network.parameters(), lr=config.lr, weight_decay=1e-4)
    lr_sched  = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.n_epochs)

    best_loss = float("inf")
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Precompute tiled stats for the flattened obs window (T_obs * obs_dim,)
    obs_mean_flat = np.tile(stats["obs_mean"], config.T_obs)   # (T_obs * obs_dim,)
    obs_std_flat  = np.maximum(np.tile(stats["obs_std"], config.T_obs), 1e-6)
    act_mean      = stats["action_mean"]                        # (action_dim,)
    act_std       = np.maximum(stats["action_std"], 1e-6)

    policy.network.train()
    for epoch in trange(config.n_epochs, desc="Training", dynamic_ncols=True):
        batch = buffer.sample(config.batch_size, config.T_obs, config.T_pred)

        obs_np = (batch["obs"].numpy() - obs_mean_flat) / obs_std_flat

        act_np   = batch["actions"].numpy()
        B, T, D  = act_np.shape
        act_np   = ((act_np.reshape(-1, D) - act_mean) / act_std).reshape(B, T, D)

        batch = {
            "obs":     torch.tensor(obs_np, dtype=torch.float32, device=device),
            "actions": torch.tensor(act_np, dtype=torch.float32, device=device),
        }

        optimizer.zero_grad()
        loss = policy.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.network.parameters(), 1.0)
        optimizer.step()
        lr_sched.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            policy.save(out_path)

        if (epoch + 1) % args.log_every == 0:
            print(f"  epoch {epoch + 1}/{config.n_epochs}  "
                  f"loss={loss_val:.5f}  best={best_loss:.5f}  "
                  f"lr={lr_sched.get_last_lr()[0]:.2e}")

    print(f"\nBest checkpoint saved → {out_path}  (loss={best_loss:.5f})")


if __name__ == "__main__":
    main()
