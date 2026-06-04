"""Evaluation metrics."""

from __future__ import annotations

import torch


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)
) -> list[torch.Tensor]:
    """Compute top-k accuracy (in percent) for each k in ``topk``.

    Args:
        output: Logits of shape ``(N, C)``.
        target: Ground-truth labels of shape ``(N,)``.
        topk: The k values to evaluate, e.g. ``(1, 5)``.

    Returns:
        One scalar tensor per entry in ``topk``, each a percentage in [0, 100].
    """
    with torch.no_grad():
        batch_size = target.size(0)
        if batch_size == 0:
            return [torch.tensor(0.0) for _ in topk]

        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
