"""Deterministic seeding for reproducible runs.

Seeds Python's ``random``, NumPy, and PyTorch (CPU + CUDA). Optionally
enables PyTorch's deterministic algorithms so that, where supported, repeated
runs on the same hardware produce identical results.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Seed all relevant RNGs.

    Args:
        seed: The seed value applied to ``random``, NumPy, and PyTorch.
        deterministic: If ``True``, request deterministic cuDNN/algorithm
            behaviour. This can slow things down and some ops have no
            deterministic implementation, so it is off by default.

    Notes:
        MPS (Apple Silicon) reproducibility depends on the PyTorch version;
        ``torch.manual_seed`` covers the global generator either way.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
