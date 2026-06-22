"""Diffusion Policy — inference and training interface.

Implements action-chunking: the policy predicts T_pred steps ahead,
and the runner executes only the first T_exec of them before re-querying.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from diluri_sim.policy.network import TemporalUNet
from diluri_sim.policy.scheduler import DDPMScheduler

if TYPE_CHECKING:
    from diluri_sim.config import DiffusionPolicyConfig


class DiffusionPolicy:
    """Joint-space Diffusion Policy for a single FR3 arm.

    Observation  : (T_obs, obs_dim) sliding window of joint pos + vel
    Action output: (T_pred, action_dim) joint position targets
    Execution    : first T_exec actions are fed to the PositionController

    Usage (inference):
        policy = DiffusionPolicy.load("checkpoint.pt")
        policy.network.eval()
        for each sim step:
            policy.update_obs(get_robot_obs(robot))
            if action_queue is empty:
                actions = policy.predict_ddim()  # (T_exec, action_dim)
                action_queue.extend(actions)
            ctrl.set_target(action_queue.popleft())

    Usage (training):
        loss = policy.compute_loss(batch)
        loss.backward()
    """

    def __init__(self, config: "DiffusionPolicyConfig") -> None:
        self.config = config
        self.network = TemporalUNet(
            action_dim=config.action_dim,
            obs_dim=config.T_obs * config.obs_dim,
            down_channels=config.down_channels,
            time_emb_dim=config.time_emb_dim,
        )
        self.scheduler = DDPMScheduler(
            n_steps=config.n_diffusion_steps,
            schedule=config.beta_schedule,
        )
        self.obs_buffer: deque[np.ndarray] = deque(maxlen=config.T_obs)

        # Optional normalization stats (set after computing from data)
        self.obs_mean:    np.ndarray | None = None
        self.obs_std:     np.ndarray | None = None
        self.action_mean: np.ndarray | None = None
        self.action_std:  np.ndarray | None = None

    # ------------------------------------------------------------------
    # Observation buffer
    # ------------------------------------------------------------------

    def update_obs(self, obs: np.ndarray) -> None:
        """Append one observation step (shape: (obs_dim,)) to the sliding window."""
        self.obs_buffer.append(obs.astype(np.float32))

    def reset(self) -> None:
        """Clear observation buffer (call at episode start)."""
        self.obs_buffer.clear()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, device: str = "cpu") -> np.ndarray:
        """Full DDPM inference (slow). Returns (T_exec, action_dim)."""
        obs = self._get_obs_tensor(device)
        x   = torch.randn(1, self.config.T_pred, self.config.action_dim, device=device)

        for t in range(self.config.n_diffusion_steps - 1, -1, -1):
            t_batch  = torch.tensor([t], device=device)
            eps      = self.network(x, t_batch, obs)
            x        = self.scheduler.ddpm_step(x, t, eps)

        return self._to_actions(x)

    def predict_ddim(
        self,
        n_steps: int | None = None,
        device: str = "cpu",
    ) -> np.ndarray:
        """Fast DDIM inference. Returns (T_exec, action_dim)."""
        if n_steps is None:
            n_steps = self.config.n_ddim_steps
        obs       = self._get_obs_tensor(device)
        x         = torch.randn(1, self.config.T_pred, self.config.action_dim, device=device)
        timesteps = self.scheduler.ddim_timesteps(n_steps)

        for i, t in enumerate(timesteps):
            t_prev  = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch = torch.tensor([t], device=device)
            eps     = self.network(x, t_batch, obs)
            x       = self.scheduler.ddim_step(x, t, t_prev, eps)

        return self._to_actions(x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """MSE loss on predicted noise.

        batch keys:
          'obs'    : (B, T_obs * obs_dim)  — normalized observations
          'actions': (B, T_pred, action_dim) — normalized actions in [-1, 1]
        """
        obs     = batch["obs"]
        actions = batch["actions"]
        device  = actions.device
        B       = actions.shape[0]

        t     = torch.randint(0, self.scheduler.n_steps, (B,), device=device)
        noise = torch.randn_like(actions)
        noisy, _ = self.scheduler.add_noise(actions, t, noise)
        eps_pred = self.network(noisy, t, obs)
        return F.mse_loss(eps_pred, noise)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def set_normalization_stats(
        self,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ) -> None:
        self.obs_mean    = obs_mean.astype(np.float32)
        self.obs_std     = np.maximum(obs_std,    1e-6).astype(np.float32)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std  = np.maximum(action_std, 1e-6).astype(np.float32)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mean is None:
            return obs
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mean is None:
            return action
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mean is None:
            return action
        return action * self.action_std + self.action_mean

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({
            "config":      self.config,
            "network":     self.network.state_dict(),
            "obs_mean":    self.obs_mean,
            "obs_std":     self.obs_std,
            "action_mean": self.action_mean,
            "action_std":  self.action_std,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DiffusionPolicy":
        ckpt   = torch.load(path, map_location=device, weights_only=False)
        policy = cls(ckpt["config"])
        policy.network.load_state_dict(ckpt["network"])
        policy.network.to(device)
        if ckpt.get("obs_mean") is not None:
            policy.set_normalization_stats(
                ckpt["obs_mean"], ckpt["obs_std"],
                ckpt["action_mean"], ckpt["action_std"],
            )
        return policy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs_tensor(self, device: str) -> torch.Tensor:
        """Stack obs_buffer to (1, T_obs * obs_dim), padding if needed."""
        obs_list = list(self.obs_buffer)
        if not obs_list:
            raise RuntimeError("obs_buffer is empty; call update_obs() before predict()")
        while len(obs_list) < self.config.T_obs:
            obs_list.insert(0, obs_list[0])
        obs = np.stack(obs_list, axis=0)   # (T_obs, obs_dim)
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
        return torch.tensor(obs.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

    def _to_actions(self, x: torch.Tensor) -> np.ndarray:
        """Denormalize and return first T_exec actions as numpy."""
        actions = x.squeeze(0).cpu().numpy()[:self.config.T_exec]
        return self.denormalize_action(actions)
