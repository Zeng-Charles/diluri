from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraConfig:
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fov: float


@dataclass(frozen=True)
class SceneConfig:
    dt: float
    substeps: int
    viewer: bool
    backend: str
    camera: CameraConfig
    horizon_steps: int


@dataclass(frozen=True)
class ControllerConfig:
    type: str                    # "position" | "velocity" | "impedance" | "force"
    params: tuple[tuple, ...]    # ((key, value), ...) — immutable, use dict(cfg.params) to read


@dataclass(frozen=True)
class RobotConfig:
    name: str
    role: str
    asset: Path
    base_pos: tuple[float, float, float]
    base_euler: tuple[float, float, float]
    fixed_base: bool
    home_q: tuple[float, ...]
    controller: ControllerConfig


@dataclass(frozen=True)
class DualFr3Config:
    scene: SceneConfig
    robots: tuple[RobotConfig, ...]


def _find_project_root(config_path: Path) -> Path:
    """Walk up from config_path to find the directory containing pyproject.toml."""
    candidate = config_path.parent
    while candidate != candidate.parent:
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError(
        f"Cannot locate project root: no pyproject.toml found above {config_path}. "
        "Run scripts from within the diluri-sim project directory."
    )


def load_dual_fr3_config(path: str | Path) -> DualFr3Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    root = _find_project_root(config_path)
    scene_raw = raw["scene"]
    camera_raw = scene_raw["camera"]
    robots = []

    for name, robot_raw in raw["robots"].items():
        asset = Path(robot_raw["asset"]).expanduser()
        if not asset.is_absolute():
            asset = root / asset

        robots.append(
            RobotConfig(
                name=name,
                role=robot_raw["role"],
                asset=asset.resolve(),
                base_pos=_triple(robot_raw["base_pos"], f"{name}.base_pos"),
                base_euler=_triple(robot_raw["base_euler"], f"{name}.base_euler"),
                fixed_base=bool(robot_raw.get("fixed_base", True)),
                home_q=tuple(float(v) for v in robot_raw["home_q"]),
                controller=_parse_controller(robot_raw.get("controller", {}), name),
            )
        )

    return DualFr3Config(
        scene=SceneConfig(
            dt=float(scene_raw["dt"]),
            substeps=int(scene_raw["substeps"]),
            viewer=bool(scene_raw["viewer"]),
            backend=str(scene_raw["backend"]),
            camera=CameraConfig(
                pos=_triple(camera_raw["pos"], "scene.camera.pos"),
                lookat=_triple(camera_raw["lookat"], "scene.camera.lookat"),
                fov=float(camera_raw["fov"]),
            ),
            horizon_steps=int(scene_raw["horizon_steps"]),
        ),
        robots=tuple(robots),
    )


def validate_assets(config: DualFr3Config) -> None:
    missing = [robot.asset for robot in config.robots if not robot.asset.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing robot asset files. Add the Franka Research 3 URDF/MJCF assets "
            "or update configs/env/dual_fr3.yaml:\n"
            f"{formatted}"
        )


def _parse_controller(raw: dict, robot_name: str) -> ControllerConfig:
    ctrl_type = raw.get("type", "position")
    valid = {"position", "velocity", "cartesian", "impedance"}
    if ctrl_type not in valid:
        raise ValueError(
            f"{robot_name}.controller.type must be one of {sorted(valid)}, got {ctrl_type!r}"
        )
    params = tuple((k, v) for k, v in raw.items() if k != "type")
    return ControllerConfig(type=ctrl_type, params=params)


# ---------------------------------------------------------------------------
# Diffusion Policy config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiffusionPolicyConfig:
    # Observation / action sizes
    obs_dim: int                          # per-step observation size (e.g. 14 = 7q + 7dq)
    action_dim: int                       # action size (e.g. 7 joint positions)
    # Horizons
    T_obs: int = 2                        # observation context window
    T_pred: int = 16                      # prediction horizon (UNet sequence length)
    T_exec: int = 8                       # executed chunk size (action chunking)
    # Diffusion
    n_diffusion_steps: int = 100
    beta_schedule: str = "cosine"         # "linear" | "cosine"
    n_ddim_steps: int = 10               # steps for fast DDIM inference
    # Network
    down_channels: tuple[int, ...] = (256, 512, 1024)
    time_emb_dim: int = 256
    # Training
    lr: float = 1e-4
    batch_size: int = 256
    n_epochs: int = 500


def load_diffusion_policy_config(path: str | Path) -> DiffusionPolicyConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    net = raw.get("network", {})
    trn = raw.get("training", {})
    dif = raw.get("diffusion", {})

    return DiffusionPolicyConfig(
        obs_dim=int(raw["obs_dim"]),
        action_dim=int(raw["action_dim"]),
        T_obs=int(raw.get("T_obs", 2)),
        T_pred=int(raw.get("T_pred", 16)),
        T_exec=int(raw.get("T_exec", 8)),
        n_diffusion_steps=int(dif.get("n_steps", 100)),
        beta_schedule=str(dif.get("beta_schedule", "cosine")),
        n_ddim_steps=int(dif.get("n_ddim_steps", 10)),
        down_channels=tuple(int(c) for c in net.get("down_channels", [256, 512, 1024])),
        time_emb_dim=int(net.get("time_emb_dim", 256)),
        lr=float(trn.get("lr", 1e-4)),
        batch_size=int(trn.get("batch_size", 256)),
        n_epochs=int(trn.get("n_epochs", 500)),
    )


def _triple(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three numbers.")
    return (float(value[0]), float(value[1]), float(value[2]))
