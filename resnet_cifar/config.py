"""Configuration settings and hyperparameters for training ResNet on CIFAR-10.

Following the paper "Deep Residual Learning for Image Recognition" by He et al.
"""

# ResNet-6n+2 CIFAR architecture parameters
# n determines the depth: depth = 6n + 2
# Typical values: n=3 -> 20 layers, n=5 -> 32 layers, n=7 -> 44 layers, n=9 -> 56 layers
N = 3

# Training Hyperparameters
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Seed for reproducibility
SEED = 42
