from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .fsq import FSQQuantizer


class Encoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), # 28x28 -> 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), # 14x14 -> 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), # 7x7 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4, 0, 1), # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, 0, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(self, levels: List[int], beta: float) -> None:
        super().__init__()
        embed_dim = len(levels)
        self.encoder = Encoder(embed_dim)
        self.decoder = Decoder(embed_dim)
        self.quantizer = FSQQuantizer(levels)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z_e = self.encoder(x)
        z_q, indices, code_usage = self.quantizer(z_e)
        x_hat = self.decoder(z_q)

        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q.detach())
        loss = recon_loss + self.beta * commit_loss

        metrics = {
            "loss": loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "code_usage": code_usage,
            "codebook_size": torch.tensor(self.quantizer.codebook_size, device=x.device),
            "indices": indices,
        }
        return x_hat, metrics
