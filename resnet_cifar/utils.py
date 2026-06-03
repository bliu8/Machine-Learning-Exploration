"""Utility functions for seed initialization and metrics calculation.

Contains helper functions for reproducible seed setting and accuracy metrics calculation.
"""

import random
import numpy as np
import torch

def set_seed(seed: int):
    """Set random seed for reproducibility across random, numpy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Note: MPS reproducibility can depend on specific torch versions,
    # but torch.manual_seed covers the random generator.

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): Model output logits of shape (N, C).
        target (torch.Tensor): Ground truth labels of shape (N,).
        topk (tuple): Tuple of integers specifying the k values (e.g. (1, 5)).

    Returns:
        list: List of float tensors representing accuracy in percentage for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if batch_size == 0:
            return [torch.tensor([0.0]) for _ in topk]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
