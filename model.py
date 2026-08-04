"""GPT with RoPE, RMSNorm, no biases, tied embeddings.

Kept deliberately close to nanoGPT so the parameter math stays predictable:
every block is exactly 12*n_embd^2 + 2*n_embd parameters.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig, embedding_params, non_embedding_params


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x is (B, n_head, T, head_dim). Rotation runs in fp32 for stability under fp16."""
    dtype = x.dtype
    x = x.float()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, : x.size(2), :]
    s = sin[None, None, : x.size(2), :]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    return out.to(dtype)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos, sin, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # On Turing (T4) SDPA falls back to the memory-efficient or math kernel.
        # Flash attention needs sm80+, so do not expect it here.
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init on residual projections, as in GPT-2.
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _apply(self, *args, **kwargs):
        """Restore weight tying after anything that rebuilds the parameters.

        nn.Module._apply can only update a parameter in place when the result is
        shallow-copy compatible with the original. A move to a different device
        type is not, so it constructs a fresh Parameter for each entry in each
        module's dict, and a Parameter shared by two modules comes back as two
        independent tensors.

        On CUDA this never surfaced. On XLA it does: .to(xla_device) silently
        untied wte from lm_head, giving 211.78M parameters where the preset says
        173.14M, an extra embedding's worth of optimizer state, and two copies
        that then train apart. The quiet part is the checkpoint, because
        load_resume drops lm_head.weight whenever the config says tied, so
        resuming one of those would have thrown away the learned output head and
        kept training as if nothing happened.

        Retying here rather than at each call site means a move cannot undo it.
        """
        out = super()._apply(*args, **kwargs)
        if self.cfg.tie_embeddings and self.lm_head.weight is not self.wte.weight:
            self.lm_head.weight = self.wte.weight
        return out

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        T = idx.size(1)
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.drop(self.wte(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)

        if targets is None:
            return self.lm_head(x[:, -1:, :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss

    def param_report(self) -> str:
        counted = sum(p.numel() for p in self.parameters())
        emb = self.wte.weight.numel() * (1 if self.cfg.tie_embeddings else 2)
        non_emb = counted - emb
        predicted = non_embedding_params(self.cfg)
        lines = [
            f"model: {self.cfg.n_layer}L / {self.cfg.n_embd}d / {self.cfg.n_head}H "
            f"(head_dim {self.cfg.head_dim}), block_size {self.cfg.block_size}",
            f"non-embedding params : {non_emb:,} ({non_emb / 1e6:.2f}M)",
            f"embedding params     : {emb:,} ({emb / 1e6:.2f}M){' tied' if self.cfg.tie_embeddings else ''}",
            f"total params         : {counted:,} ({counted / 1e6:.2f}M)",
        ]
        if non_emb != predicted:
            lines.append(f"WARNING: analytic count {predicted:,} disagrees with the live count")
        return "\n".join(lines)

    def configure_optimizer(self, lr: float, weight_decay: float, betas: tuple[float, float], device_type: str):
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs = {}
        if device_type == "cuda":
            kwargs["fused"] = True  # supported on Turing, meaningfully faster
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, **kwargs)

    def estimated_flops_per_token(self) -> float:
        n = sum(p.numel() for p in self.parameters())
        attn = 12 * self.cfg.n_layer * self.cfg.n_embd * self.cfg.block_size
        return 6 * n + attn


def strip_prefixes(state: dict) -> dict:
    """Remove DDP and torch.compile wrappers from state dict keys."""
    out = {}
    for k, v in state.items():
        for prefix in ("module.", "_orig_mod."):
            while k.startswith(prefix):
                k = k[len(prefix) :]
        out[k] = v
    return out


def load_model_from_checkpoint(path, device="cpu", override_block_size: int | None = None) -> GPT:
    path = Path(path)
    if path.is_dir():
        raise SystemExit(
            f"{path} is a directory, not a checkpoint file.\n"
            "An interrupted download often leaves a folder named like the file. "
            "Delete it and download the .pt again.")
    if not path.exists():
        raise SystemExit(f"no such checkpoint: {path}")
    if path.stat().st_size < 1024:
        raise SystemExit(
            f"{path} is only {path.stat().st_size} bytes, so the download did not finish. "
            "Delete it and try again.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["config"]["model"] if "model" in ckpt.get("config", {}) else ckpt["config"])
    if override_block_size:
        cfg_dict["block_size"] = override_block_size
    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg)
    state = strip_prefixes(ckpt["model"])
    if cfg.tie_embeddings:
        state.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("rope_")]
    missing = [k for k in missing if not k.startswith("rope_") and k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch. missing={missing} unexpected={unexpected}")
    return model.to(device).eval().float(), ckpt


if __name__ == "__main__":
    cfg = GPTConfig()
    print(GPT(cfg).param_report())
