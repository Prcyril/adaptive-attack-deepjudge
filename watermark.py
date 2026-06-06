import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# CIFAR-10 mean / std for un-normalising   (provided in codebase.setup)
from codebase import setup


def train(
        ae: nn.Module,
        decoder: nn.Module,
        wm: np.ndarray,
        dset: torch.utils.data.Dataset,
        exp_cfg,
) -> None:


    # hyperparamaters
    batch_size   = 128
    n_epochs     = 25          
    lr           = 1e-3
    weight_decay = 1e-5
    lam_mse      = 0.80        
    clamp_eps    = 0.01        
    clean_reps   = 5          
    

    device = exp_cfg.device
    ae, decoder = ae.to(device), decoder.to(device)

    loader = DataLoader(
        dset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )

    opt = torch.optim.AdamW(
        list(ae.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    bce_bits = nn.BCEWithLogitsLoss()
    mse_img  = nn.MSELoss()

    # watermark bits as float tensor  (shape = (k,))
    wm_bits = torch.tensor(wm, dtype=torch.float32, device=device)
    k = wm_bits.numel()

    # tensors for (un)normalising CIFAR-10 images
    cifar_mean = torch.tensor(setup.CIFAR10_MEAN, dtype=torch.float32,
                          device=device).view(1, 3, 1, 1)
    cifar_std  = torch.tensor(setup.CIFAR10_STD , dtype=torch.float32,
                          device=device).view(1, 3, 1, 1)
    for epoch in range(1, n_epochs + 1):
        ae.train(); decoder.train()
        sum_loss = sum_bce = sum_mse = 0.0

        for clean_x, _ in loader:
            clean_x = clean_x.to(device)                     
            # create watermarked images via auto-encoder
            wm_x = ae(clean_x)

            # clamp perturbation in TRUE pixel space 
            clean_pix = clean_x * cifar_std + cifar_mean
            wm_pix    = wm_x   * cifar_std + cifar_mean
            delta     = (wm_pix - clean_pix).clamp(-clamp_eps, clamp_eps)
            wm_pix    = (clean_pix + delta).clamp(0.0, 1.0)
            wm_x      = (wm_pix - cifar_mean) / cifar_std    # back to network space

            #  build a mixed batch: many clean + few watermarked
            x_mix   = torch.cat([clean_x.repeat(clean_reps, 1, 1, 1), wm_x], dim=0)
            tgt_mix = torch.cat([
                torch.zeros(clean_reps * clean_x.size(0), k, device=device),
                wm_bits.repeat(clean_x.size(0), 1)
            ], dim=0)

            #  decoder prediction + losses
            logits   = decoder(x_mix)            # ( (clean_reps+1)*N , k )
            loss_bce = bce_bits(logits, tgt_mix)
            loss_mse = mse_img(wm_x, clean_x)
            loss     = loss_bce + lam_mse * loss_mse

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_loss += loss.item()
            sum_bce  += loss_bce.item()
            sum_mse  += loss_mse.item()

        print(f"Epoch {epoch:02d} | "
              f"loss={sum_loss/len(loader):.4f}  "
              f"BCE={sum_bce/len(loader):.4f}  "
              f"MSE={sum_mse/len(loader):.6f}")
