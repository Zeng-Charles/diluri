from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from diluri_sim.config import ControllerConfig

# FR3 joint-space defaults (7 joints)
_KP     = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0])
_KV     = np.array([ 450.0,  450.0,  350.0,  350.0,  200.0,  200.0,  200.0])
_FLIMIT = np.array([  87.0,   87.0,   87.0,   87.0,   12.0,   12.0,   12.0])


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RobotController(ABC):
    def __init__(self, robot, arm_dofs: int = 7) -> None:
        self._robot = robot
        self._arm_dofs = arm_dofs
        self._dof_idx = list(range(arm_dofs))

    @abstractmethod
    def set_target(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    @property
    def robot(self):
        return self._robot


# ---------------------------------------------------------------------------
# Position controller — wraps Genesis control_dofs_position (built-in PD)
# ---------------------------------------------------------------------------

class PositionController(RobotController):
    """Joint-space position controller using Genesis built-in PD control."""

    def __init__(
        self,
        robot,
        kp: np.ndarray | None = None,
        kv: np.ndarray | None = None,
        force_limits: np.ndarray | None = None,
        arm_dofs: int = 7,
    ) -> None:
        super().__init__(robot, arm_dofs)
        n = arm_dofs
        kp  = _KP[:n].copy()     if kp           is None else np.asarray(kp,           dtype=float)
        kv  = _KV[:n].copy()     if kv           is None else np.asarray(kv,           dtype=float)
        lim = _FLIMIT[:n].copy() if force_limits is None else np.asarray(force_limits, dtype=float)
        robot.set_dofs_kp(kp,  dofs_idx_local=self._dof_idx)
        robot.set_dofs_kv(kv,  dofs_idx_local=self._dof_idx)
        robot.set_dofs_force_range(-lim, lim, dofs_idx_local=self._dof_idx)
        self._target: torch.Tensor = robot.get_dofs_position(dofs_idx_local=self._dof_idx).clone()

    def set_target(self, q_des) -> None:
        self._target = torch.as_tensor(q_des, dtype=torch.float32)

    def step(self) -> None:
        self._robot.control_dofs_position(self._target, dofs_idx_local=self._dof_idx)


# ---------------------------------------------------------------------------
# Velocity controller — wraps Genesis control_dofs_velocity (built-in)
# ---------------------------------------------------------------------------

class VelocityController(RobotController):
    """Joint-space velocity controller using Genesis built-in velocity control."""

    def __init__(
        self,
        robot,
        kv: np.ndarray | None = None,
        force_limits: np.ndarray | None = None,
        arm_dofs: int = 7,
    ) -> None:
        super().__init__(robot, arm_dofs)
        n = arm_dofs
        kv  = _KV[:n].copy()     if kv           is None else np.asarray(kv,           dtype=float)
        lim = _FLIMIT[:n].copy() if force_limits is None else np.asarray(force_limits, dtype=float)
        robot.set_dofs_kp(np.zeros(n), dofs_idx_local=self._dof_idx)
        robot.set_dofs_kv(kv,          dofs_idx_local=self._dof_idx)
        robot.set_dofs_force_range(-lim, lim, dofs_idx_local=self._dof_idx)
        self._target: torch.Tensor = torch.zeros(n)

    def set_target(self, dq_des) -> None:
        self._target = torch.as_tensor(dq_des, dtype=torch.float32)

    def step(self) -> None:
        self._robot.control_dofs_velocity(self._target, dofs_idx_local=self._dof_idx)


# ---------------------------------------------------------------------------
# Cartesian controller — Genesis IK + built-in position control
# ---------------------------------------------------------------------------

class CartesianController(RobotController):
    """Cartesian position controller: Genesis inverse_kinematics → control_dofs_position.

    Replaces custom Cartesian impedance math with Genesis built-in IK.
    Requires the robot to be added with ``requires_jac_and_IK=True``.
    """

    def __init__(
        self,
        robot,
        ee_link_name: str = "fr3_link8",
        kp: np.ndarray | None = None,
        kv: np.ndarray | None = None,
        force_limits: np.ndarray | None = None,
        arm_dofs: int = 7,
    ) -> None:
        super().__init__(robot, arm_dofs)
        n = arm_dofs
        kp  = _KP[:n].copy()     if kp           is None else np.asarray(kp,           dtype=float)
        kv  = _KV[:n].copy()     if kv           is None else np.asarray(kv,           dtype=float)
        lim = _FLIMIT[:n].copy() if force_limits is None else np.asarray(force_limits, dtype=float)
        robot.set_dofs_kp(kp,  dofs_idx_local=self._dof_idx)
        robot.set_dofs_kv(kv,  dofs_idx_local=self._dof_idx)
        robot.set_dofs_force_range(-lim, lim, dofs_idx_local=self._dof_idx)
        self._ee_link = robot.get_link(ee_link_name)
        idx = [self._ee_link.idx_local]
        self._target_pos  = robot.get_links_pos(idx).squeeze(0).clone()
        self._target_quat = robot.get_links_quat(idx).squeeze(0).clone()

    def set_target(self, pos: np.ndarray | None = None, quat: np.ndarray | None = None) -> None:
        if pos is not None:
            self._target_pos = torch.as_tensor(pos, dtype=torch.float32)
        if quat is not None:
            self._target_quat = torch.as_tensor(quat, dtype=torch.float32)

    def step(self) -> None:
        q_des = self._robot.inverse_kinematics(
            link=self._ee_link,
            pos=self._target_pos,
            quat=self._target_quat,
        )
        self._robot.control_dofs_position(
            q_des[..., :self._arm_dofs],
            dofs_idx_local=self._dof_idx,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_controller(robot, cfg: "ControllerConfig") -> RobotController:
    t = cfg.type
    p = dict(cfg.params)

    if t == "position":
        return PositionController(
            robot,
            kp=_arr(p, "kp"),
            kv=_arr(p, "kv"),
            force_limits=_arr(p, "force_limits"),
        )
    if t == "velocity":
        return VelocityController(
            robot,
            kv=_arr(p, "kv"),
            force_limits=_arr(p, "force_limits"),
        )
    if t in ("cartesian", "impedance"):
        return CartesianController(
            robot,
            ee_link_name=p.get("ee_link", "fr3_link8"),
            kp=_arr(p, "kp"),
            kv=_arr(p, "kv"),
            force_limits=_arr(p, "force_limits"),
        )
    raise ValueError(f"Unknown controller type: {t!r}. Valid: position, velocity, cartesian.")


def _arr(p: dict, key: str) -> np.ndarray | None:
    v = p.get(key)
    return None if v is None else np.asarray(v, dtype=float)
