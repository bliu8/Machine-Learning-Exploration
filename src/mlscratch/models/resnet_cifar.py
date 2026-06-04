"""ResNet for CIFAR-10 (the 6n+2 family from He et al., Section 4.2).

Depth is ``6n + 2``:
  - 1 initial 3x3 conv (16 filters)
  - 3 stages of ``n`` :class:`~mlscratch.layers.residual.BasicBlock` blocks
    (each block = 2 conv layers) -> ``6n`` layers
  - 1 final fully-connected layer

Example depths: ``n=3`` -> 20, ``n=5`` -> 32, ``n=7`` -> 44, ``n=9`` -> 56.

The forward pass is intentionally left unimplemented for now.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mlscratch.layers.residual import BasicBlock


def depth_for(n: int) -> int:
    """Return the total layer depth ``6n + 2`` for a given ``n``."""
    return 6 * n + 2


class ResNetCifar(nn.Module):
    """CIFAR-10 ResNet of depth ``6n + 2``.

    Stages (channels, spatial size with 32x32 input):
      - Stage 1: ``n`` blocks, 16 channels, 32x32
      - Stage 2: ``n`` blocks, 32 channels, 16x16 (first block stride 2)
      - Stage 3: ``n`` blocks, 64 channels, 8x8 (first block stride 2)
    """

    def __init__(self, n: int, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = self._make_stage(16, n, stride=1)
        self.stage2 = self._make_stage(32, n, stride=2)
        self.stage3 = self._make_stage(64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: relu(bn1(conv1(x))) -> stage1 -> stage2 -> stage3 ->
        #       avgpool -> flatten -> fc
        raise NotImplementedError("ResNetCifar.forward is not implemented yet.")
