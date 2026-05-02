from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torchvision.utils import save_image

from .data import get_mnist_loaders
from .model import VQVAE
from .utils import get_device


def parse_levels(levels: str) -> List[int]:
    return [int(value) for value in levels.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render reconstructions and print used codes")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--levels", type=str, default="5,5,5,5")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-path", type=str, default="outputs/recon_codes.png")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--max-codes", type=int, default=200)
    args = parser.parse_args()

    device = get_device(args.device)
    levels = parse_levels(args.levels)

    model = VQVAE(levels=levels, beta=0.25).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    _, test_loader = get_mnist_loaders(args.batch_size, args.num_workers, args.data_dir)

    all_codes = []
    images = []
    recons = []

    with torch.no_grad():
        for batch_index, (batch, _) in enumerate(test_loader):
            if batch_index >= args.max_batches:
                break
            batch = batch.to(device)
            x_hat, metrics = model(batch)
            images.append(batch.cpu())
            recons.append(x_hat.cpu())

            indices = metrics["indices"].reshape(-1).cpu().tolist()
            all_codes.extend(indices)

    if not images:
        raise RuntimeError("No batches were loaded from the dataset.")

    inputs = torch.cat(images, dim=0)
    outputs = torch.cat(recons, dim=0)
    grid = torch.cat([inputs, outputs], dim=0)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path, nrow=inputs.size(0), normalize=True, value_range=(-1, 1))

    unique_codes = sorted(set(all_codes))
    if args.max_codes > 0:
        unique_codes = unique_codes[: args.max_codes]

    print(f"Unique codes used (count={len(unique_codes)}):")
    print(",".join(str(code) for code in unique_codes))


if __name__ == "__main__":
    main()
