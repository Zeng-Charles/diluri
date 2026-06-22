"""DDPM / DDIM noise schedulers for Diffusion Policy.

References:
  - Ho et al., 2020 (DDPM): https://arxiv.org/abs/2006.11239
  - Song et al., 2020 (DDIM): https://arxiv.org/abs/2010.02502
  - Chi et al., 2023 (Diffusion Policy): https://arxiv.org/abs/2303.04137
"""
from __future__ import annotations

import numpy as np
import torch


def _make_betas(n_steps: int, schedule: str) -> np.ndarray:
    if schedule == "linear":
        return np.linspace(1e-4, 2e-2, n_steps)
    if schedule == "cosine":
        t = np.linspace(0, n_steps, n_steps + 1) / n_steps
        alphas_bar = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
        return np.clip(betas, 0.0, 0.999)
    raise ValueError(f"Unknown beta schedule: {schedule!r}. Use 'linear' or 'cosine'.")


class DDPMScheduler:
    """Denoising Diffusion Probabilistic Model scheduler.

    Handles both training (add_noise) and inference (ddpm_step / ddim_step).
    """

    def __init__(self, n_steps: int = 100, schedule: str = "cosine") -> None:
        self.n_steps = n_steps
        betas = _make_betas(n_steps, schedule)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)

        self.betas       = torch.tensor(betas,      dtype=torch.float32)
        self.alphas      = torch.tensor(alphas,     dtype=torch.float32)
        self.alphas_bar  = torch.tensor(alphas_bar, dtype=torch.float32)
        self._sqrt_ab    = torch.sqrt(self.alphas_bar)
        self._sqrt_1m_ab = torch.sqrt(1.0 - self.alphas_bar)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion q(x_t | x_0): return (x_t, noise)."""
        if noise is None:
            noise = torch.randn_like(x0)
        B = x0.shape[0]
        shape = (B,) + (1,) * (x0.ndim - 1)
        sqrt_ab   = self._sqrt_ab[t].to(x0.device).view(shape)
        sqrt_1m_ab = self._sqrt_1m_ab[t].to(x0.device).view(shape)
        return sqrt_ab * x0 + sqrt_1m_ab * noise, noise

    # ------------------------------------------------------------------
    # Inference — DDPM (full T steps)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddpm_step(
        self,
        x_t: torch.Tensor,
        t: int,
        eps_theta: torch.Tensor,
    ) -> torch.Tensor:
        """Single DDPM reverse step p(x_{t-1} | x_t)."""
        dev  = x_t.device
        beta = self.betas[t].to(dev)
        alph = self.alphas[t].to(dev)
        ab   = self.alphas_bar[t].to(dev)

        # μ_θ = (1/√α_t) * (x_t − β_t/√(1−ᾱ_t) * ε_θ)
        coef = beta / torch.sqrt(1.0 - ab)
        mu   = (x_t - coef * eps_theta) / torch.sqrt(alph)

        if t > 0:
            sigma = torch.sqrt(beta)
            return mu + sigma * torch.randn_like(x_t)
        return mu

    # ------------------------------------------------------------------
    # Inference — DDIM (fewer steps, deterministic when eta=0)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        eps_theta: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single DDIM reverse step."""
        dev    = x_t.device
        ab     = self.alphas_bar[t].to(dev)
        ab_prev = self.alphas_bar[t_prev].to(dev) if t_prev >= 0 else torch.tensor(1.0, device=dev)

        # Predict x_0 from current estimate
        x0_pred = (x_t - torch.sqrt(1.0 - ab) * eps_theta) / torch.sqrt(ab)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab) * (1.0 - ab / ab_prev))
        direction = torch.sqrt((1.0 - ab_prev - sigma ** 2).clamp(min=0.0)) * eps_theta
        noise = sigma * torch.randn_like(x_t) if eta > 0.0 else 0.0
        return torch.sqrt(ab_prev) * x0_pred + direction + noise

    def ddim_timesteps(self, n_steps: int) -> list[int]:
        """Uniformly spaced timestep schedule for DDIM inference.

        Starts at t=step_size-1 so the highest timestep (t=n_steps-1, pure noise)
        is always included. E.g. n_steps=10, T=100 → [99,89,...,9].
        """
        step_size = max(self.n_steps // n_steps, 1)
        ts = list(range(step_size - 1, self.n_steps, step_size))
        return list(reversed(ts[:n_steps]))
