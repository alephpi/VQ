from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import torch
from torchvision.utils import save_image

from .model import VQVAE
from .utils import get_device


def parse_levels(levels: str) -> List[int]:
    return [int(value) for value in levels.split(",") if value.strip()]


def _parse_code(code: str, levels: List[int]) -> int:
    if "," in code:
        parts = [int(value.strip()) for value in code.split(",") if value.strip()]
        if len(parts) != len(levels):
            raise ValueError("Code must have the same number of entries as levels.")
        index = 0
        multiplier = 1
        for value, level in zip(parts, levels):
            if value < 0 or value >= level:
                raise ValueError("Code entries must be within [0, level-1].")
            index += value * multiplier
            multiplier *= level
        return index
    index = int(code)
    if index < 0:
        raise ValueError("Code index must be >= 0.")
    codebook_size = int(math.prod(levels))
    if index >= codebook_size:
        raise ValueError("Code index must be < codebook_size.")
    return index


def render_codebook_grid(
    checkpoint_path: Path,
    output_path: Path,
    levels: List[int],
    device: torch.device,
    input_size: int,
) -> None:
    model = VQVAE(levels=levels, beta=0.25).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    codebook_size = int(math.prod(levels))
    height = input_size // 4
    width = input_size // 4

    indices = torch.arange(codebook_size, device=device, dtype=torch.long)
    indices = indices.view(codebook_size, 1, 1).expand(codebook_size, height, width)
    z_q = model.quantizer.from_index(indices)

    with torch.no_grad():
        decoded = model.decoder(z_q)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nrow = int(math.prod(levels[:2]))
    if nrow <= 0 or codebook_size % nrow != 0:
        nrow = int(math.sqrt(codebook_size))
    save_image(decoded, output_path, nrow=nrow, normalize=True, value_range=(-1, 1))


def render_single_code(
    checkpoint_path: Path,
    output_path: Path,
    levels: List[int],
    device: torch.device,
    input_size: int,
    code: str,
) -> None:
    model = VQVAE(levels=levels, beta=0.25).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    index = _parse_code(code, levels)
    height = input_size // 4
    width = input_size // 4

    indices = torch.full((1, height, width), index, device=device, dtype=torch.long)
    z_q = model.quantizer.from_index(indices)

    with torch.no_grad():
        decoded = model.decoder(z_q)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(decoded, output_path, normalize=True, value_range=(-1, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render all FSQ codes into a grid image")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--levels", type=str, default="5,5,5,5")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-size", type=int, default=28)
    parser.add_argument("--output-path", type=str, default="outputs/codebook_grid.png")
    parser.add_argument(
        "--code",
        type=str,
        default="",
        help="Render a single code (global index or comma-separated digits).",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    levels = parse_levels(args.levels)
    if args.code:
        render_single_code(
            checkpoint_path=Path(args.checkpoint),
            output_path=Path(args.output_path),
            levels=levels,
            device=device,
            input_size=args.input_size,
            code=args.code,
        )
    else:
        render_codebook_grid(
            checkpoint_path=Path(args.checkpoint),
            output_path=Path(args.output_path),
            levels=levels,
            device=device,
            input_size=args.input_size,
        )


if __name__ == "__main__":
    main()
