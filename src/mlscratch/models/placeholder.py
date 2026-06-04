"""A trivial, fully-working model.

This exists so the test suite and benchmark harness run end-to-end before the
real architectures (e.g. :class:`~mlscratch.models.resnet_cifar.ResNetCifar`)
are implemented. It is a stand-in for sanity-checking plumbing, not a model
worth training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyConvNet(nn.Module):
    """A minimal conv -> pool -> linear classifier for 32x32x3 inputs."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
