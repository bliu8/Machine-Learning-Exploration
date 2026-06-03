"""ResNet model architecture for CIFAR-10 (6n+2 variant).

Implements the architecture described in "Deep Residual Learning for Image Recognition"
by He et al. (Section 4.2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaLayer(nn.Module):
    """A helper layer to run user-defined functions or lambda expressions in nn.Sequential."""
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    """Basic residual block for CIFAR-10 ResNet.

    Consists of two 3x3 convolutions, each followed by Batch Normalization
    and ReLU activation, with a residual shortcut connection.
    """
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        # First 3x3 conv with optional stride for downsampling
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # Second 3x3 conv (always stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut mapping
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            # Option-A zero-padding shortcut (parameter-free)
            # Spatially downsamples by stride using slicing or pooling,
            # and pads the channel dimension with zeros to match output channels.
            # TODO: Initialize/implement Option-A zero-padding shortcut.
            # Hint: You can use a LambdaLayer that does F.pad and downsamples.
            # Example structure:
            # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, 0, planes - in_planes)))
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for BasicBlock.
        # Steps:
        # 1. out = relu(bn1(conv1(x)))
        # 2. out = bn2(conv2(out))
        # 3. out = out + shortcut(x)
        # 4. out = relu(out)
        raise NotImplementedError("BasicBlock forward pass is not implemented yet.")

class ResNetCifar(nn.Module):
    """ResNet model variant for CIFAR-10.

    Total depth is 6n + 2 layers:
    - 1 initial convolutional layer (16 filters)
    - 3 stages of n residual blocks each (each block has 2 convolutional layers): 3 * 2 * n = 6n layers
    - 1 final fully-connected linear layer

    Total depth configuration examples:
    - n = 3 -> 20 layers
    - n = 5 -> 32 layers
    - n = 7 -> 44 layers
    - n = 9 -> 56 layers

    Spatial dimensions and channel sizes:
    - Stage 1: n blocks, input/output channels = 16, spatial size = 32x32
    - Stage 2: n blocks, input = 16, output = 32, spatial size = 16x16 (stride 2 on first block)
    - Stage 3: n blocks, input = 32, output = 64, spatial size = 8x8 (stride 2 on first block)
    """
    def __init__(self, n: int, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16

        # Initial layer: 3x3 conv with 16 filters
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Three stages of n residual blocks
        self.stage1 = self._make_layer(16, n, stride=1)
        self.stage2 = self._make_layer(32, n, stride=2)
        self.stage3 = self._make_layer(64, n, stride=2)

        # Global average pooling and final linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Helper to create a stage of ResNet residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for ResNetCifar.
        # Steps:
        # 1. out = relu(bn1(conv1(x)))
        # 2. out = stage1(out)
        # 3. out = stage2(out)
        # 4. out = stage3(out)
        # 5. out = avgpool(out)
        # 6. out = out.view(out.size(0), -1)
        # 7. out = fc(out)
        raise NotImplementedError("ResNetCifar forward pass is not implemented yet.")
