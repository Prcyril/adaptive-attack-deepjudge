import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from codebase.datasets.poisoned import PoisonedDataset
from codebase import setup


def train(
        bad_module: nn.Module,
        poison_target: int,
        trigger_size: int,
        trigger_alpha: float,
        exp_cfg,
) -> None:
    device = exp_cfg.device
    num_epochs = 10
    batch_size = 128
    lr = 1e-3

    # Define red 5x5 trigger
    trigger = torch.zeros((trigger_size, trigger_size, 3), dtype=torch.uint8)
    trigger[:, :, 0] = 255
    trigger_loc = [32 - trigger_size, 32 - trigger_size]

    # Shared transform 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(setup.CIFAR10_MEAN, setup.CIFAR10_STD),
    ])

    # Base CIFAR10 clean dataset 
    clean_train_set = torchvision.datasets.CIFAR10(
        root=str(exp_cfg.data_dir), 
        train=True, 
        download=True, 
        transform=transform
    )

    # Poisoned dataset
    poisoned_train_set = PoisonedDataset(
        clean_dset=clean_train_set,
        poison_rate=0.5,
        poison_target=poison_target,
        trigger=trigger.numpy(),
        trigger_alpha=trigger_alpha,
        trigger_loc=trigger_loc,
        poison_seed=42,
    )

    # DataLoader 
    train_loader = DataLoader(poisoned_train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # Optimizer and loss
    optimizer = optim.Adam(bad_module.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop 
    bad_module.train()
    bad_module.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Determine which samples are poisoned 
            is_poisoned = (labels == poison_target)
            clean_mask = ~is_poisoned  

            optimizer.zero_grad()
            outputs = bad_module(images)

            loss = criterion(outputs, labels)

            # L2 penalty to suppress backdoor on clean inputs
            if clean_mask.any():
                loss += 0.1 * torch.norm(outputs[clean_mask], p=2)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss:.4f}")