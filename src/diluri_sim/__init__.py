"""Simulation package for dual-arm ultrasound-guided robotic intervention."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

__version__ = "0.1.0"


def configure_runtime() -> None:
    """Set env vars and strip user-site from sys.path before importing Genesis."""
    cache = Path("/tmp/diluri-sim")
    os.environ.setdefault("NUMBA_CACHE_DIR",  str(cache / "numba"))
    os.environ.setdefault("MPLCONFIGDIR",     str(cache / "matplotlib"))
    user_site = site.getusersitepackages()
    if user_site in sys.path:
        sys.path.remove(user_site)
