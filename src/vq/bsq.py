import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySphericalQuantizer(nn.Module):
    """
    Binary Spherical Quantizer (BSQ)
    输入 z: (B, D) 或 (B, T, D)
    输出: 与输入同形，值域 {-1/√D, +1/√D}（单位球面上的顶点）
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 归一化因子：使量化后向量仍在单位球面上
        self.register_buffer("scale", torch.tensor(dim).sqrt().reciprocal())

    def _bits_to_indices(self, bits: torch.Tensor) -> torch.Tensor:
        basis = torch.pow(2, torch.arange(self.dim - 1, -1, -1, device=bits.device))
        return (bits.to(torch.long) * basis).sum(dim=-1)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """纯量化，无梯度技巧"""
        return torch.sign(z) * self.scale  # 二值化，保持在球面上

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, p=2, dim=-1)   # 投影到单位球
        z_q = self.quantize(z)
        z_q_st = z + (z_q - z).detach()
        commit_loss = F.mse_loss(z, z_q.detach())

        bits = (z_q > 0).to(torch.uint8)
        indices = self._bits_to_indices(bits)
        flat_indices = indices.reshape(-1)
        unique_indices, counts = torch.unique(flat_indices, return_counts=True)
        used_indices_probs = counts.float() / flat_indices.numel()
        entropy = -(used_indices_probs * torch.log(used_indices_probs + 1e-10)).sum()
        perplexity = torch.exp(entropy)

        info_dict = {
            "indices": indices,
            "perplexity": perplexity,
            "commit_loss": commit_loss,
        }
        return z_q_st, info_dict

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """返回 {0, 1} 的二进制码（便于存储/索引）"""
        z_norm = F.normalize(z, p=2, dim=-1)
        return (torch.sign(z_norm) > 0).to(torch.uint8)  # (B, D) bool

    def decode(self, bits: torch.Tensor) -> torch.Tensor:
        """从 {0,1} 码还原量化向量"""
        return (bits.float() * 2 - 1) * self.scale


# ── 快速验证 ──────────────────────────────────────────────
if __name__ == "__main__":
    D = 64
    bsq = BinarySphericalQuantizer(dim=D)

    z = torch.randn(4, D, requires_grad=True)        # batch=4
    z_q, info = bsq(z)

    print(f"输入范数: {z.norm(dim=-1).mean():.3f}")
    print(f"输出范数: {z_q.norm(dim=-1).mean():.3f}")  # ≈ 1.0
    print(f"Commit loss: {info['commit_loss'].item():.4f}")

    # 验证梯度能反传
    info["commit_loss"].backward()
    print(f"梯度存在: {z.grad is not None}")

    # 编解码往返
    bits = bsq.encode(z)
    z_rec = bsq.decode(bits)
    print(f"编解码误差: {(z_rec - bsq.quantize(z)).abs().max().item():.6f}")  # ≈ 0