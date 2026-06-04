"""Smoke tests: the package imports, seeding is deterministic, and the
placeholder model does a forward + backward pass on synthetic input.
"""

from __future__ import annotations

import torch

from mlscratch import set_seed
from mlscratch.models import TinyConvNet, depth_for
from mlscratch.utils import accuracy


def test_seed_is_deterministic():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_depth_formula():
    assert depth_for(3) == 20
    assert depth_for(9) == 56


def test_placeholder_forward_backward():
    set_seed(0)
    model = TinyConvNet(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    target = torch.randint(0, 10, (4,))

    logits = model(x)
    assert logits.shape == (4, 10)

    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_accuracy_perfect_and_zero():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    assert accuracy(logits, torch.tensor([1, 0]))[0].item() == 100.0
    assert accuracy(logits, torch.tensor([0, 1]))[0].item() == 0.0
