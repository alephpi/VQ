from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torchvision.utils import save_image

import logging
# Configure logger
logger = logging.getLogger("BSQ")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)


def get_logger() -> logging.Logger:
    return logger



def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def save_reconstructions(path: Path, inputs: torch.Tensor, recons: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([inputs, recons], dim=0)
    save_image(grid, path, nrow=inputs.size(0), normalize=True, value_range=(-1, 1))
