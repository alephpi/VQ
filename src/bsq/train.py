from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import get_mnist_loaders
from .model import VQVAE
from .utils import get_device, save_checkpoint, save_reconstructions, set_seed


def parse_levels(levels: str) -> List[int]:
    return [int(value) for value in levels.split(",") if value.strip()]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            _, metrics = model(batch)
            total_loss += metrics["loss"].item() * batch.size(0)
            count += batch.size(0)
    return torch.tensor(total_loss / max(count, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VQ-VAE with FSQ or BSQ on MNIST")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--levels", type=str, default="5,5,5")
    parser.add_argument("--quantizer", type=str, default="fsq", choices=["fsq", "bsq"])
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    levels = parse_levels(args.levels)
    device = get_device(args.device)
    set_seed(args.seed)

    train_loader, test_loader = get_mnist_loaders(
        args.batch_size, args.num_workers, args.data_dir
    )

    model = VQVAE(
        levels=levels,
        beta=args.beta,
        quantizer=args.quantizer,
        codebook_size=args.codebook_size,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # total_steps = args.epochs * len(train_loader)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=total_steps,
    #     eta_min=1e-5,
    # )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch, _ in progress:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, metrics = model(batch)
            loss = metrics["loss"]
            loss.backward()
            optimizer.step()
            # scheduler.step()
            progress.set_postfix(
                loss=f"{metrics['loss'].item():.4f}",
                recon=f"{metrics['recon_loss'].item():.4f}",
                commit=f"{metrics['commit_loss'].item():.4f}",
                perplexity=f"{metrics['perplexity']:.3f}",
                usage=f"{metrics['code_usage']:.3f}",
            )

        test_loss = evaluate(model, test_loader, device)
        sample_inputs, _ = next(iter(test_loader))
        sample_inputs = sample_inputs[:8].to(device)
        with torch.no_grad():
            sample_recons, _ = model(sample_inputs)

        save_reconstructions(
            run_dir / f"recon_epoch_{epoch:03d}.png",
            sample_inputs.cpu(),
            sample_recons.cpu(),
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: test_loss={test_loss.item():.4f}, lr={current_lr:.6f}")

    save_checkpoint(run_dir / f"checkpoint_epoch_{args.epochs:03d}.pt", model, optimizer, args.epochs)


if __name__ == "__main__":
    main()
