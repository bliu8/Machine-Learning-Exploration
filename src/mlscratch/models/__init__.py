"""Model architectures."""

from mlscratch.models.placeholder import TinyConvNet
from mlscratch.models.resnet_cifar import ResNetCifar, depth_for

__all__ = ["ResNetCifar", "TinyConvNet", "depth_for"]
