# diluri-sim

Genesis simulation scaffold for dual-arm ultrasound-guided robotic intervention research.

Two Franka Research 3 arms operate in a shared workspace: one manipulates an ultrasound probe, the other a needle. The package includes a **Diffusion Policy** implementation (Chi et al., 2023) for learning joint-space manipulation from demonstrations.

## Project Layout

```text
.
├── assets/
│   └── robots/franka_fr3/          # FR3 URDF assets
├── configs/
│   ├── env/
│   │   └── dual_fr3.yaml           # Scene geometry, robot placement, controllers
│   └── policy/
│       └── diffusion.yaml          # Diffusion Policy hyperparameters
├── scripts/
│   ├── visualize.py                # Launch scene viewer (no policy)
│   ├── collect.py                  # Collect demonstration episodes
│   ├── train.py                    # Train Diffusion Policy
│   └── eval.py                     # Run trained policy in simulation
├── src/
│   └── diluri_sim/
│       ├── config.py               # YAML config dataclasses + loaders
│       ├── sim/                    # Genesis simulation (scene, controllers)
│       ├── policy/                 # Diffusion Policy (network, scheduler, inference)
│       └── dataset/                # Replay buffer for demonstration storage
└── tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # includes pytest
```

`genesis-world` has heavy native dependencies (Taichi, optional CUDA). If your platform requires a custom Genesis build, install Genesis first and then run `pip install -e ".[dev]"`.

## Robot Assets

Place the Franka Research 3 URDF under:

```text
assets/robots/franka_fr3/fr3.urdf
```

The asset path can be changed in `configs/env/dual_fr3.yaml`.

## Workflow

### 1 — Visualise the scene

```bash
python scripts/visualize.py
python scripts/visualize.py --headless --steps 200   # headless smoke-check
```

### 2 — Collect demonstrations

Runs a sinusoidal scan trajectory on the left arm and saves episodes to a replay buffer:

```bash
python scripts/collect.py --output data/demos.pkl --episodes 20
python scripts/collect.py --output data/demos.pkl --episodes 20 --headless
```

### 3 — Train Diffusion Policy

```bash
python scripts/train.py \
    --buffer  data/demos.pkl \
    --output  checkpoints/diffusion_policy.pt
```

Optional flags: `--device cuda`, `--log-every 100`.

### 4 — Evaluate trained policy

```bash
python scripts/eval.py --policy checkpoints/diffusion_policy.pt
python scripts/eval.py --policy checkpoints/diffusion_policy.pt --headless --steps 500
```

## Configuration

| File | Controls |
| --- | --- |
| `configs/env/dual_fr3.yaml` | Scene physics (dt, substeps, backend), robot base poses, home joint angles, controller gains |
| `configs/policy/diffusion.yaml` | Observation/action dimensions, prediction horizon, DDPM/DDIM steps, network architecture, training hyperparameters |

## Architecture

**Diffusion Policy** (Chi et al., 2023) with a Temporal UNet backbone:

- Observation: joint positions + velocities, `T_obs=2` step window → (28,) vector
- Action: joint position targets, predicted `T_pred=16` steps ahead
- Execution: action chunking — first `T_exec=8` steps executed before re-querying
- Inference: DDIM with 10 steps (10× faster than full DDPM)
- Network: ~25 M parameters (1D Conv residual blocks, FiLM conditioning, GroupNorm)

## Current Scope

Simulation only. The following are not yet implemented: teleoperation, perception, tissue model, ultrasound model, needle model.
