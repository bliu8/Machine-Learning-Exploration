# CIFAR-10 ResNet-6n+2 Classifier Setup

This repository contains the scaffolded setup for a CIFAR-10 classifier implementing the ResNet architecture variant described in Section 4.2 of ["Deep Residual Learning for Image Recognition" (He et al.)](https://arxiv.org/abs/1512.03385).

## Architecture Details

For CIFAR-10, the paper designs a specific ResNet architecture family of depth $6n + 2$ using parameter-free **Option-A zero-padding shortcuts**.

- **Initial layer**: Single $3 \times 3$ convolution with 16 filters.
- **Stage 1**: $n$ blocks of $3 \times 3$ convolutions with 16 filters (output size: $32 \times 32$).
- **Stage 2**: $n$ blocks of $3 \times 3$ convolutions with 32 filters (output size: $16 \times 16$). The first block downsamples spatial size via stride=2.
- **Stage 3**: $n$ blocks of $3 \times 3$ convolutions with 64 filters (output size: $8 \times 8$). The first block downsamples spatial size via stride=2.
- **Pooling & Output**: Global average pooling, followed by a 10-way fully-connected layer.

### Depth Table

| $n$ parameter | Total Layer Depth ($6n + 2$) |
|:-------------:|:----------------------------:|
|       3       |              20              |
|       5       |              32              |
|       7       |              44              |
|       9       |              56              |

---

## Directory Structure

```
resnet_cifar/
  ├── __init__.py
  ├── config.py    # Hyperparameters: n, batch_size, epochs, lr, weight_decay, momentum
  ├── data.py      # CIFAR-10 DataLoaders (fully implemented with augmentations)
  ├── model.py     # BasicBlock + ResNetCifar (signatures + TODO stubs)
  ├── train.py     # Training and evaluation loop skeleton (with TODO stubs)
  └── utils.py     # set_seed() and accuracy helper (fully implemented)
requirements.txt   # Pinned dependencies
README.md          # Setup and architectural documentation
smoke_test.py      # Verification script
```

---

## Installation & Setup

1. **Create the virtual environment** (requires Python 3.11+):
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the environment**:
   - On macOS/Linux:
     ```bash
     source .venv/bin/pipe/activate  # or source .venv/bin/activate
     ```
   - On Windows:
     ```cmd
     .venv\Scripts\activate
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Verification

To run the verification smoke test which downloads the CIFAR-10 dataset, configures loaders, pulls one batch, verifies sizes, and checks model initialization:

```bash
python smoke_test.py
```
