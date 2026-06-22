"""Replay buffer for storing and sampling demonstration episodes."""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import torch


class ReplayBuffer:
    """Stores demonstration episodes as (obs, action) sequences.

    Each episode is a dict with:
      'obs'    : np.ndarray (L, obs_dim)
      'actions': np.ndarray (L, action_dim)

    Sampling returns sliding windows:
      obs window    : (T_obs, obs_dim)  — the T_obs steps before an action
      action window : (T_pred, action_dim) — T_pred steps starting from that action
    """

    def __init__(self) -> None:
        self.episodes: list[dict[str, np.ndarray]] = []
        self._current: dict[str, list[np.ndarray]] = {"obs": [], "actions": []}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add(self, obs: np.ndarray, action: np.ndarray) -> None:
        """Append one transition to the current episode."""
        self._current["obs"].append(obs.astype(np.float32))
        self._current["actions"].append(action.astype(np.float32))

    def end_episode(self) -> None:
        """Finalise the current episode and start a new one."""
        if self._current["obs"]:
            self.episodes.append({
                "obs":     np.array(self._current["obs"],     dtype=np.float32),
                "actions": np.array(self._current["actions"], dtype=np.float32),
            })
        self._current = {"obs": [], "actions": []}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        T_obs: int,
        T_pred: int,
    ) -> dict[str, torch.Tensor]:
        """Sample a batch of (flattened obs window, action window) pairs.

        Returns:
          obs    : (B, T_obs * obs_dim)
          actions: (B, T_pred, action_dim)
        """
        obs_batch: list[np.ndarray] = []
        act_batch: list[np.ndarray] = []
        min_len = T_obs + T_pred

        # Filter usable episodes once
        usable = [ep for ep in self.episodes if len(ep["obs"]) >= min_len]
        if not usable:
            raise ValueError(
                f"No episodes long enough to sample T_obs={T_obs} + T_pred={T_pred} steps. "
                f"Longest episode: {max((len(e['obs']) for e in self.episodes), default=0)} steps."
            )

        while len(obs_batch) < batch_size:
            ep    = random.choice(usable)
            L     = len(ep["obs"])
            start = random.randint(0, L - min_len)
            obs_win = ep["obs"][start: start + T_obs]                      # (T_obs, obs_dim)
            act_win = ep["actions"][start + T_obs: start + T_obs + T_pred] # (T_pred, act_dim)
            obs_batch.append(obs_win.flatten())
            act_batch.append(act_win)

        return {
            "obs":     torch.tensor(np.array(obs_batch), dtype=torch.float32),
            "actions": torch.tensor(np.array(act_batch), dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Normalisation stats
    # ------------------------------------------------------------------

    def compute_stats(self) -> dict[str, np.ndarray]:
        """Return obs/action mean and std across all episodes."""
        all_obs     = np.concatenate([ep["obs"]     for ep in self.episodes], axis=0)
        all_actions = np.concatenate([ep["actions"] for ep in self.episodes], axis=0)
        return {
            "obs_mean":    all_obs.mean(axis=0),
            "obs_std":     all_obs.std(axis=0),
            "action_mean": all_actions.mean(axis=0),
            "action_std":  all_actions.std(axis=0),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.episodes, f)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        buf = cls()
        with open(path, "rb") as f:
            buf.episodes = pickle.load(f)
        return buf

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(ep["obs"]) for ep in self.episodes)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(episodes={len(self.episodes)}, "
            f"transitions={len(self)})"
        )
