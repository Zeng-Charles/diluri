#!/usr/bin/env python
"""Collect demonstration episodes using a fixed sinusoidal scan trajectory.

Joint 0 (shoulder pan) sweeps ±0.4 rad around home — mimicking a probe scan.
Each episode adds small Gaussian noise so the policy learns to handle
slight perturbations rather than memorising one exact trajectory.

Usage:
    python scripts/collect.py --output data/demos.pkl
    python scripts/collect.py --output data/demos.pkl --episodes 30 --headless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diluri_sim import configure_runtime
configure_runtime()

from diluri_sim.config import load_dual_fr3_config
from diluri_sim.sim import build_scene, build_controller
from diluri_sim.dataset import ReplayBuffer

ARM_DOFS = 7

# ---------------------------------------------------------------------------
# Trajectory definition
# ---------------------------------------------------------------------------

# TRAJ_STEPS / SCAN_AMP / NOISE_STD are task constants; HOME_Q is read from
# config at runtime so it stays in sync with configs/env/dual_fr3.yaml.
TRAJ_STEPS = 300    # one complete scan ≈ 3 s at dt=0.01
SCAN_AMP   = 0.4    # rad
NOISE_STD  = 0.005  # per-step Gaussian noise on joint targets (rad)


def make_trajectory(home_q: np.ndarray, noise_std: float = NOISE_STD) -> np.ndarray:
    """Return joint targets of shape (TRAJ_STEPS, 7).

    Joint 0 follows a full sine cycle around home_q; the rest stay at home + small noise.
    """
    t    = np.linspace(0, 2 * np.pi, TRAJ_STEPS, dtype=np.float32)
    traj = np.tile(home_q, (TRAJ_STEPS, 1))
    traj[:, 0] += SCAN_AMP * np.sin(t).astype(np.float32)
    traj += np.random.normal(0, noise_std, traj.shape).astype(np.float32)
    return traj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_robot_obs(robot) -> np.ndarray:
    """Joint positions + velocities → (14,) observation vector."""
    idx = list(range(ARM_DOFS))
    q  = robot.get_dofs_position(dofs_idx_local=idx).cpu().numpy()
    dq = robot.get_dofs_velocity(dofs_idx_local=idx).cpu().numpy()
    return np.concatenate([q, dq]).astype(np.float32)


def reset_to_home(robot, ctrl, scene, home_q: np.ndarray, n_settle: int = 150) -> None:
    """Teleport joints to home_q, clear velocity, then settle under PD control."""
    idx = list(range(ARM_DOFS))
    robot.set_dofs_position(home_q.copy(), dofs_idx_local=idx)
    robot.set_dofs_velocity(np.zeros(ARM_DOFS, dtype=np.float32), dofs_idx_local=idx)
    ctrl.set_target(home_q.copy())
    for _ in range(n_settle):
        ctrl.step()
        scene.step()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect sinusoidal scan demonstrations.")
    p.add_argument("--config",   default=str(ROOT / "configs/env/dual_fr3.yaml"))
    p.add_argument("--output",   default=str(ROOT / "data/demos.pkl"))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    sim_config = load_dual_fr3_config(args.config)
    scene, robots = build_scene(sim_config, show_viewer=not args.headless)

    target_name = "left_probe_arm"
    robot       = robots[target_name]
    robot_cfg   = next(r for r in sim_config.robots if r.name == target_name)
    home_q      = np.array(robot_cfg.home_q, dtype=np.float32)  # source of truth from config

    controllers = {
        rc.name: build_controller(robots[rc.name], rc.controller)
        for rc in sim_config.robots
    }
    ctrl = controllers[target_name]

    buffer = ReplayBuffer()

    for ep in range(args.episodes):
        # Reset to home smoothly (teleport + velocity clear + PD settle)
        reset_to_home(robot, ctrl, scene, home_q)

        # Each episode gets a slightly different noisy version of the trajectory
        traj = make_trajectory(home_q)   # (TRAJ_STEPS, 7)

        for step_idx in range(TRAJ_STEPS):
            obs    = get_robot_obs(robot)
            action = traj[step_idx]          # joint position target

            buffer.add(obs, action)

            ctrl.set_target(action)
            for c in controllers.values():
                c.step()

            try:
                scene.step()
            except Exception:
                break

        buffer.end_episode()
        q_final = get_robot_obs(robot)[:ARM_DOFS]
        print(f"  ep {ep + 1:>2}/{args.episodes}  "
              f"joint0_final={q_final[0]:+.3f} rad  "
              f"(target {traj[-1, 0]:+.3f})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    buffer.save(out)
    print(f"\nSaved {buffer} → {out}")
    print(f"obs shape per step : (14,)  [7 q + 7 dq]")
    print(f"action shape per step: (7,)   [joint position targets]")


if __name__ == "__main__":
    main()
