"""Residual building blocks for CIFAR-style ResNets.

Implements the architecture described in "Deep Residual Learning for Image
Recognition" (He et al., Section 4.2). The forward passes are intentionally
left unimplemented for now.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class LambdaLayer(nn.Module):
    """Wrap an arbitrary function so it can live inside ``nn.Sequential``."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class BasicBlock(nn.Module):
    """Basic residual block: two 3x3 convs, each with BatchNorm and ReLU.

    A residual shortcut connects input to output. When the spatial size or
    channel count changes, the paper's parameter-free Option-A shortcut
    (stride downsampling + zero-padding on the channel dimension) is used.
    """

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            # TODO: Option-A zero-padding shortcut (parameter-free).
            # Downsample spatially by `stride` and zero-pad the channel dim
            # from `in_planes` to `planes`, e.g. via a LambdaLayer + F.pad.
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: out = relu(bn1(conv1(x))); out = bn2(conv2(out));
        #       out = relu(out + shortcut(x))
        raise NotImplementedError("BasicBlock.forward is not implemented yet.")
