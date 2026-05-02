from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


class FSQQuantizer(nn.Module):
    def __init__(self, levels: List[int]) -> None:
        super().__init__()
        if not levels or any(l < 2 for l in levels):
            raise ValueError("All levels must be >= 2.")
        levels_tensor = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", levels_tensor)
        self.register_buffer("steps", 2.0 / (levels_tensor - 1.0))
        self.codebook_size = int(torch.prod(levels_tensor).item())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tanh(x)
        view_shape = [1, -1] + [1] * (x.ndim - 2)
        levels = self.levels.view(*view_shape)
        steps = self.steps.view(*view_shape)

        x_scaled = (x + 1.0) / steps
        min_levels = torch.zeros_like(levels)
        x_rounded = torch.round(x_scaled).clamp(min=min_levels, max=levels - 1.0)
        x_quant = x_rounded * steps - 1.0

        x_st = x + (x_quant - x).detach()
        indices = self._codes_from_rounded(x_rounded)
        perplexity = self._perplexity(indices)
        code_usage = self._code_usage(indices)
        return x_st, indices, code_usage

    def _codes_from_rounded(self, x_rounded: torch.Tensor) -> torch.Tensor:
        if x_rounded.ndim == 4:
            b, c, h, w = x_rounded.shape
            flat = x_rounded.permute(0, 2, 3, 1).reshape(-1, c)
            codes = self._flat_codes(flat)
            return codes.view(b, h, w)
        if x_rounded.ndim == 2:
            return self._flat_codes(x_rounded)
        raise ValueError("Expected a 2D or 4D tensor for quantization.")

    def _flat_codes(self, flat: torch.Tensor) -> torch.Tensor:
        levels = self.levels.to(dtype=torch.long)
        multipliers = torch.cumprod(
            torch.cat([torch.ones(1, device=levels.device, dtype=levels.dtype), levels[:-1]]),
            dim=0,
        )
        flat_long = flat.to(dtype=torch.long)
        return (flat_long * multipliers.view(1, -1)).sum(dim=-1)

    def _perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.tensor(0.0, device=indices.device)
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.codebook_size).float()
        probs = counts / counts.sum()
        entropy = -(probs * (probs + 1e-8).log()).sum()
        return entropy.exp()

    def _code_usage(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.tensor(0.0, device=indices.device)
        flat = indices.reshape(-1)
        unique = torch.unique(flat)
        return unique.numel() / float(self.codebook_size)

    def from_index(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices 的逆操作，从整数索引还原量化后的特征值。

        Args:
            indices: (B, H, W) 或 (N,)  整数 codebook 索引
        Returns:
            (B, C, H, W) 或 (N, C)  量化值，值域 [-1, 1]
        """
        if indices.ndim == 3:
            b, h, w = indices.shape
            flat = indices.reshape(-1)                                   # (B*H*W,)
            x_rounded = self._rounded_from_flat_codes(flat)              # (B*H*W, C)
            x_rounded = x_rounded.view(b, h, w, -1).permute(0, 3, 1, 2) # (B, C, H, W)
            return self._rounded_to_quant(x_rounded)

        if indices.ndim == 1:
            x_rounded = self._rounded_from_flat_codes(indices)           # (N, C)
            return self._rounded_to_quant(x_rounded)

        raise ValueError("Expected 1D or 3D indices tensor.")

    def _rounded_from_flat_codes(self, flat: torch.Tensor) -> torch.Tensor:
        """_flat_codes 的逆：(N,) → (N, C)，混合进制拆解"""
        levels = self.levels.to(dtype=torch.long)
        multipliers = torch.cumprod(
            torch.cat([torch.ones(1, device=levels.device, dtype=levels.dtype), levels[:-1]]),
            dim=0,
        )                                                  # [1, L0, L0*L1, ...]
        # (N, 1) // (1, C) % (1, C)  →  每维整数值 [0, Li-1]
        x_rounded = (flat.unsqueeze(-1) // multipliers.view(1, -1)) % levels.view(1, -1)
        return x_rounded.float()

    def _rounded_to_quant(self, x_rounded: torch.Tensor) -> torch.Tensor:
        """整数值 [0, Li-1] → 量化值 [-1, 1]，与 forward 中 x_quant 完全对称"""
        view_shape = [1, -1] + [1] * (x_rounded.ndim - 2)
        steps = self.steps.view(*view_shape)
        return x_rounded * steps - 1.0