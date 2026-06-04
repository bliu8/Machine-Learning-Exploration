# mlscratch

A personal project for implementing foundational machine-learning techniques
from scratch in PyTorch, with an emphasis on readable code, reproducible runs,
and honest benchmarking.

The first target is a CIFAR-10 ResNet — the 6n+2 family from
["Deep Residual Learning for Image Recognition" (He et al.)](https://arxiv.org/abs/1512.03385),
Section 4.2. The architecture is scaffolded; the residual forward passes are
still being filled in.

## Layout

```
src/mlscratch/
  layers/      # reusable blocks (e.g. residual BasicBlock)
  models/      # architectures (ResNetCifar; TinyConvNet placeholder)
  utils/       # seeding (utils/seed.py) and metrics
benchmarks/    # timing + peak-memory harness -> Markdown table
tests/         # pytest suite
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"           # or: make install
```

Dependencies are pinned in [`pyproject.toml`](pyproject.toml).

## Usage

```bash
make test      # run the pytest suite
make bench     # run the benchmark harness (writes benchmarks/results.md)
make lint      # ruff + black --check
make format    # ruff --fix + black
```

The benchmark harness measures median forward and forward+backward latency
(and peak memory on CUDA) on synthetic CIFAR-shaped input, and appends a row
to `benchmarks/results.md`:

```bash
python benchmarks/bench.py --model tiny --batch-size 128 --iters 20
```

## Reproducibility

- `mlscratch.utils.set_seed(seed, deterministic=...)` seeds Python, NumPy,
  and PyTorch (CPU + CUDA). Pass `deterministic=True` to request PyTorch's
  deterministic algorithms and disable cuDNN autotuning — this trades some
  speed for run-to-run repeatability on the same hardware.
- Dependencies are pinned.
- Note: exact bitwise reproducibility is only guaranteed on the same hardware,
  drivers, and library versions; MPS (Apple Silicon) determinism varies by
  PyTorch version.

## Results

_To be added once the model is trained._
