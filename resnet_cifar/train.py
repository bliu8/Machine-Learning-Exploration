"""Training script skeleton for CIFAR-10 ResNet classifier.

Scaffolds CLI parsing, device setup, model initialization, and the outer
training and evaluation loops. Optimization steps are left as TODOs.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from resnet_cifar.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM, N, SEED
from resnet_cifar.data import get_dataloaders
from resnet_cifar.model import ResNetCifar
from resnet_cifar.utils import set_seed, accuracy

def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Scaffold for ResNet CIFAR-10 training")
    parser.add_argument("--n", type=int, default=N,
                        help="Depth parameter n (depth of model is 6n + 2)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Weight decay coefficient")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to directory for storing/downloading data")
    args = parser.parse_args()

    # 2. Set Seed for Reproducibility
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    # 3. Device Setup
    # Uses CUDA if available, MPS (Apple Silicon GPU) if available, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 4. Data Loading
    print("Preparing CIFAR-10 data loaders...")
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )

    # 5. Model, Loss, Optimizer and Scheduler Initialization
    print(f"Building ResNet (n={args.n}, depth={6*args.n + 2})...")
    model = ResNetCifar(n=args.n, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=MOMENTUM,
        weight_decay=args.weight_decay
    )

    # Paper schedule: divide learning rate by 10 at 32k and 48k steps.
    # For batch_size=128, 50000 training examples is ~391 iterations per epoch.
    # 32k steps is ~82 epochs. 48k steps is ~123 epochs.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[82, 123],
        gamma=0.1
    )

    # 6. Training & Evaluation Loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Loop over training batches
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # TODO: Implement training forward & backward passes and optimization steps
            # 1. Zero the parameter gradients: optimizer.zero_grad()
            # 2. Forward pass: outputs = model(inputs)
            # 3. Calculate loss: loss = criterion(outputs, targets)
            # 4. Backward pass: loss.backward()
            # 5. Optimize: optimizer.step()
            # Keep track of train_loss and accuracy for statistics
            pass

        # --- Evaluation Phase ---
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        # Disable gradient calculations for evaluation
        with torch.no_grad():
            # Loop over evaluation batches
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                # TODO: Implement evaluation forward pass and validation metric aggregation
                # 1. Forward pass: outputs = model(inputs)
                # 2. Calculate loss: loss = criterion(outputs, targets)
                # Keep track of test_loss and accuracy for statistics
                pass

        # Update learning rate schedule
        # TODO: Step learning rate scheduler (if applicable)
        # scheduler.step()

        print(f"Epoch {epoch} complete.")

if __name__ == "__main__":
    main()
