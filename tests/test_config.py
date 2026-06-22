from pathlib import Path

import pytest

from diluri_sim.config import ControllerConfig, load_dual_fr3_config
from diluri_sim.env.controllers import (
    PositionController,
    VelocityController,
    CartesianController,
    build_controller,
)

CONFIG = Path(__file__).parents[1] / "configs" / "env" / "dual_fr3.yaml"


def test_scene_fields():
    cfg = load_dual_fr3_config(CONFIG)
    assert cfg.scene.dt == 0.01
    assert cfg.scene.substeps == 4
    assert cfg.scene.backend == "cpu"


def test_robot_count_and_roles():
    cfg = load_dual_fr3_config(CONFIG)
    assert len(cfg.robots) == 2
    assert {r.role for r in cfg.robots} == {"ultrasound_probe", "needle"}


def test_home_q_valid_for_fr3():
    cfg = load_dual_fr3_config(CONFIG)
    FR3_LIMITS = [
        (-2.3093,  2.3093),
        (-1.5133,  1.5133),
        (-2.4937,  2.4937),
        (-2.7478, -0.4461),
        (-2.4800,  2.4800),
        ( 0.8521,  4.2094),
        (-2.6895,  2.6895),
    ]
    for robot in cfg.robots:
        assert len(robot.home_q) == 7, f"{robot.name}: home_q must have 7 values"
        for i, (q, (lo, hi)) in enumerate(zip(robot.home_q, FR3_LIMITS)):
            assert lo <= q <= hi, (
                f"{robot.name}: joint {i+1} home_q={q:.4f} outside [{lo}, {hi}]"
            )


def test_controller_defaults():
    cfg = load_dual_fr3_config(CONFIG)
    for robot in cfg.robots:
        assert isinstance(robot.controller, ControllerConfig)
        assert robot.controller.type in {"position", "velocity", "cartesian", "impedance"}


def test_invalid_controller_type(tmp_path):
    import yaml

    raw = yaml.safe_load(CONFIG.read_text())
    raw["robots"]["left_probe_arm"]["controller"] = {"type": "unknown"}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump(raw))

    with pytest.raises(ValueError, match="controller.type"):
        load_dual_fr3_config(bad)
