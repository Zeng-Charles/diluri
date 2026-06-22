#!/usr/bin/env python
"""Run a Genesis scene interactively — no policy, controllers hold home positions.

Useful for inspecting the scene, verifying robot configurations, and testing
the viewer setup before running data collection or policy evaluation.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --headless --steps 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diluri_sim import configure_runtime
configure_runtime()

from diluri_sim.config import load_dual_fr3_config
from diluri_sim.env import build_scene, build_controller


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Genesis scene with two Franka FR3 arms.")
    p.add_argument("--config",   default=str(ROOT / "configs/env/dual_fr3.yaml"))
    p.add_argument("--headless", action="store_true", help="Disable viewer.")
    p.add_argument("--steps",    type=int, default=None, help="Steps to run (default: infinite).")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_dual_fr3_config(args.config)
    scene, robots = build_scene(config, show_viewer=not args.headless)

    controllers = {
        rc.name: build_controller(robots[rc.name], rc.controller)
        for rc in config.robots
    }

    step, steps = 0, args.steps
    while steps is None or step < steps:
        for ctrl in controllers.values():
            ctrl.step()
        try:
            scene.step()
        except Exception:
            break
        step += 1


if __name__ == "__main__":
    main()
