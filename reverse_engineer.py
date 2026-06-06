import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from codebase import setup


def reverse_trigger(
        model: nn.Module,
        target_label: int,
        exp_cfg,
) -> (np.ndarray, np.ndarray):
    
    # Set device and prepare model
    device = exp_cfg.device
    model.to(device)
    model.eval()
	
    # Hyperparameters
    batch_size = 128
    epochs = 15
    learning_rate = 0.1
    l1_lambda = 0.02

    # Setup normalization using CIFAR-10 mean and std
    cifar10_mean = setup.CIFAR10_MEAN
    cifar10_std = setup.CIFAR10_STD
    mean = torch.tensor(cifar10_mean).reshape(1, 3, 1, 1).to(device)
    std = torch.tensor(cifar10_std).reshape(1, 3, 1, 1).to(device)

     # Loading CIFAR-10 training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=str(exp_cfg.data_dir), 
        train=True, download=True, 
        transform=transform
    )
    train_indices = [i for i, label in enumerate(train_set.targets) if label != target_label]
    train_dset = Subset(train_set, train_indices)

    data_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

     # Initialize learnable trigger and mask 
    trigger_tensor = torch.randn(1, 3, 32, 32, device=device, requires_grad=True)
    mask_tensor = torch.full((1, 1, 32, 32), -10.0, device=device, requires_grad=True)

    #  Optimizer setup
    optimizer = optim.Adam([trigger_tensor, mask_tensor], lr=learning_rate)

    # Training loop to optimize the trigger and mask
    for epoch in range(epochs):
        prog_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_imgs, _ in prog_bar:
            
            batch_imgs = batch_imgs.to(device)
            batch_size = batch_imgs.size(0)
            
            # Create a batch of target labels
            labels = torch.full((batch_size,), target_label, dtype=torch.long, device=device)

            # Convert raw trigger and mask to usable range
            trigger = (torch.tanh(trigger_tensor) * 127.5 + 127.5)
            mask = torch.sigmoid(mask_tensor)

            # Un-normalize input images to uint8 space
            images_uint8 = torch.clamp(batch_imgs * std + mean, 0., 1.) * 255.
            
            # Blend the trigger into the input images using the mask
            poisoned = images_uint8 * (1 - mask) + trigger * mask
            
            # Re-normalize blended images for model input
            input_tensor = (poisoned / 255. - mean) / std

            # Forward pass and loss calculation
            logits = model(input_tensor.float())
            loss_ce = F.cross_entropy(logits, labels)
            loss_l1 = torch.norm(mask, p=1)
            loss = loss_ce + l1_lambda * loss_l1
            
            
            # Backward and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # Post-process trigger and mask for output 
    final_trigger = (torch.tanh(trigger_tensor) * 127.5 + 127.5).detach().cpu().squeeze(0)
    reversed_trigger = np.transpose(final_trigger.numpy(), (1, 2, 0)).astype(np.uint8)

    final_mask = torch.sigmoid(mask_tensor).detach().cpu().squeeze().numpy().astype(np.float32)

    return reversed_trigger, final_mask

