"""Smoke test to verify data loaders, batch shapes, and model setup.
"""

import sys
import torch

from resnet_cifar.config import BATCH_SIZE, SEED
from resnet_cifar.data import get_dataloaders
from resnet_cifar.model import ResNetCifar
from resnet_cifar.utils import set_seed

def run_smoke_test():
    print("--- Starting ResNet CIFAR-10 Smoke Test ---")
    
    # 1. Reproducibility
    set_seed(SEED)
    print(f"Reproducibility seed set to: {SEED}")

    # 2. Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Detected target device: {device}")

    # 3. Build loaders
    print("\nBuilding data loaders (this may download CIFAR-10 if not present)...")
    try:
        train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, data_dir="./data")
        print("Data loaders successfully built.")
    except Exception as e:
        print(f"FAILED to build data loaders: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Pull one training batch
    try:
        train_iter = iter(train_loader)
        images, labels = next(train_iter)
        print("\n--- Training Batch Retained ---")
        print(f"Batch image tensor shape : {images.shape} (Expected: [{BATCH_SIZE}, 3, 32, 32])")
        print(f"Batch label tensor shape : {labels.shape} (Expected: [{BATCH_SIZE}])")
        print(f"Verify images are on CPU (loaded) and ready for transfer.")
        images_device = images.to(device)
        labels_device = labels.to(device)
        print(f"Successfully transferred batch to device: {images_device.device}")
    except Exception as e:
        print(f"FAILED to fetch or process training batch: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Instantiate scaffold model
    print("\nInstantiating ResNet model scaffold (n=3)...")
    try:
        model = ResNetCifar(n=3, num_classes=10)
        print(f"Model successfully instantiated: {type(model).__name__}")
        print(f"Number of parameters in model (initialized but not trained): {sum(p.numel() for p in model.parameters()):,}")
        
        # Verify model structure
        print("\nModel Structure Summary:")
        print(model)
    except Exception as e:
        print(f"FAILED to instantiate ResNet Cifar model: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Smoke Test Passed Successfully (No training occurred) ---")

if __name__ == "__main__":
    run_smoke_test()
