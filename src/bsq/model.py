from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .bsq import BinarySphericalQuantizer
from .fsq import FiniteScalarQuantizer


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
            # nn.Conv2d(128, 256, 2, 2, 1), # 4x4 -> 2x2
            # nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 1),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(256, 128, 2, 2, 1),
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
    def __init__(
        self,
        levels: List[int],
        beta: float,
        quantizer: str = "fsq",
        codebook_size: int = 4096,
    ) -> None:
        super().__init__()
        if quantizer not in {"fsq", "bsq"}:
            raise ValueError("quantizer must be 'fsq' or 'bsq'")

        if quantizer == "fsq":
            embed_dim = len(levels)
        else:
            if codebook_size < 2 or (codebook_size & (codebook_size - 1)) != 0:
                raise ValueError("codebook_size must be a power of 2 for BSQ")
            embed_dim = int(math.log2(codebook_size))
        self.encoder = Encoder(embed_dim)
        self.decoder = Decoder(embed_dim)
        if quantizer == "fsq":
            self.quantizer = FiniteScalarQuantizer(
                input_dim=embed_dim,
                output_dim=embed_dim,
                levels=levels,
            )
            self.codebook_size = self.quantizer.all_codebook_size
        else:
            self.quantizer = BinarySphericalQuantizer(embed_dim)
            self.codebook_size = codebook_size
        self.beta = beta

        # print parameters for debugging
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {total_params} parameters.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z_e = self.encoder(x)
        batch_size, channels, height, width = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        z_q_flat, info = self.quantizer(z_e_flat)
        indices = info["indices"]
        perplexity = info["perplexity"]
        z_q = z_q_flat.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        x_hat = self.decoder(z_q)

        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = info.get("commit_loss", F.mse_loss(z_e, z_q.detach()))
        loss = recon_loss + self.beta * commit_loss

        flat_indices = indices.view(-1)
        code_usage = torch.unique(flat_indices).numel() / float(self.codebook_size)

        metrics = {
            "loss": loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "code_usage": torch.tensor(code_usage, device=x.device),
            "perplexity": perplexity,
            "codebook_size": torch.tensor(self.codebook_size, device=x.device),
            "indices": indices.reshape(batch_size, height, width),
        }
        return x_hat, metrics
