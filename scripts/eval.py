#!/usr/bin/env python
"""Run the Genesis simulation with a trained Diffusion Policy driving one arm.

The policy queries the network every T_exec steps (action chunking); between
queries, previously predicted actions are executed without re-querying.

Usage:
    python scripts/eval.py \\
        --policy checkpoints/diffusion_policy.pt \\
        [--config configs/env/dual_fr3.yaml] [--headless] [--steps 1000]
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diluri_sim import configure_runtime
configure_runtime()

from diluri_sim.config import load_dual_fr3_config
from diluri_sim.env import build_scene, build_controller
from diluri_sim.policy import DiffusionPolicy

ARM_DOFS = 7


def get_robot_obs(robot) -> np.ndarray:
    """Joint positions + velocities → (14,) observation vector."""
    dof_idx = list(range(ARM_DOFS))
    q  = robot.get_dofs_position(dofs_idx_local=dof_idx).cpu().numpy()
    dq = robot.get_dofs_velocity(dofs_idx_local=dof_idx).cpu().numpy()
    return np.concatenate([q, dq]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run simulation with Diffusion Policy.")
    p.add_argument("--config",   default=str(ROOT / "configs/env/dual_fr3.yaml"))
    p.add_argument("--policy",   required=True, help="Path to trained policy .pt checkpoint")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps",    type=int, default=None, help="Steps to run (default: infinite)")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    sim_config = load_dual_fr3_config(args.config)
    scene, robots = build_scene(sim_config, show_viewer=not args.headless)

    controllers = {
        rc.name: build_controller(robots[rc.name], rc.controller)
        for rc in sim_config.robots
    }

    # Load policy for the first robot
    target_name = sim_config.robots[0].name
    robot  = robots[target_name]
    ctrl   = controllers[target_name]

    policy = DiffusionPolicy.load(args.policy, device=args.device)
    policy.network.eval()
    print(f"Loaded policy: obs_dim={policy.config.obs_dim}  "
          f"action_dim={policy.config.action_dim}  "
          f"T_exec={policy.config.T_exec}")

    action_queue: deque[np.ndarray] = deque()
    step  = 0
    total = args.steps
    inferences = 0

    while total is None or step < total:
        obs = get_robot_obs(robot)
        policy.update_obs(obs)

        # Re-query policy when the action chunk is exhausted
        if not action_queue:
            with torch.no_grad():
                actions = policy.predict_ddim(device=args.device)   # (T_exec, action_dim)
            action_queue.extend(actions)
            inferences += 1

        action = action_queue.popleft()
        ctrl.set_target(action)

        # Step all controllers (non-policy arms hold their last target)
        for name, c in controllers.items():
            c.step()

        try:
            scene.step()
        except Exception:
            break
        step += 1

    print(f"Ran {step} sim steps, {inferences} policy queries.")


if __name__ == "__main__":
    main()
