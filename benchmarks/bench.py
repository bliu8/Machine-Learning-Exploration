"""Micro-benchmark: time forward/backward passes and record peak memory.

Runs a configurable model on synthetic CIFAR-shaped input, measures median
forward and forward+backward latency over several iterations, records peak
allocated memory (CUDA only), and appends a row to a Markdown results table.

Usage:
    python benchmarks/bench.py                 # default: TinyConvNet
    python benchmarks/bench.py --batch-size 64 --iters 100
    python benchmarks/bench.py --out benchmarks/results.md
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from mlscratch.models import TinyConvNet
from mlscratch.utils import set_seed

# Registry of benchmarkable model factories. Add real models here as their
# forward passes get implemented.
MODELS: dict[str, callable] = {
    "tiny": lambda: TinyConvNet(num_classes=10),
}

RESULTS_HEADER = (
    "| model | device | batch | fwd (ms) | fwd+bwd (ms) | peak mem (MB) |\n"
    "|---|---|---|---|---|---|\n"
)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    iters: int,
    warmup: int,
) -> dict[str, float]:
    """Return median forward and forward+backward latency, plus peak memory."""
    model = model.to(device)
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)
    criterion = torch.nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    fwd_times: list[float] = []
    full_times: list[float] = []

    for i in range(iters + warmup):
        # Forward only.
        model.eval()
        with torch.no_grad():
            _sync(device)
            t0 = time.perf_counter()
            model(x)
            _sync(device)
            t1 = time.perf_counter()

        # Forward + backward.
        model.train()
        model.zero_grad(set_to_none=True)
        _sync(device)
        t2 = time.perf_counter()
        loss = criterion(model(x), target)
        loss.backward()
        _sync(device)
        t3 = time.perf_counter()

        if i >= warmup:
            fwd_times.append((t1 - t0) * 1e3)
            full_times.append((t3 - t2) * 1e3)

    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else float("nan")
    )

    return {
        "fwd_ms": statistics.median(fwd_times),
        "full_ms": statistics.median(full_times),
        "peak_mb": peak_mb,
    }


def append_row(out_path: Path, row: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists() or out_path.read_text().strip() == "":
        out_path.write_text("# Benchmark results\n\n" + RESULTS_HEADER)
    with out_path.open("a") as f:
        f.write(row + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="tiny")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results.md"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    model = MODELS[args.model]()

    print(f"Benchmarking '{args.model}' on {device} (batch={args.batch_size})...")
    res = benchmark(model, device, args.batch_size, args.iters, args.warmup)

    peak = "n/a" if res["peak_mb"] != res["peak_mb"] else f"{res['peak_mb']:.1f}"
    row = (
        f"| {args.model} | {device.type} | {args.batch_size} "
        f"| {res['fwd_ms']:.3f} | {res['full_ms']:.3f} | {peak} |"
    )
    append_row(args.out, row)

    print(RESULTS_HEADER + row)
    print(f"\nAppended to {args.out}")


if __name__ == "__main__":
    main()
