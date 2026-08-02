"""Model and training config, plus exact non-embedding parameter accounting.

Run `python config.py 67` to see configs near a target non-embedding size.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict


@dataclass
class GPTConfig:
    n_layer: int = 14
    n_head: int = 10
    n_embd: int = 640
    block_size: int = 1024
    vocab_size: int = 50304  # 50257 GPT-2 BPE rounded up to a multiple of 64
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd {self.n_embd} not divisible by n_head {self.n_head}")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def as_dict(self) -> dict:
        return asdict(self)


# Named shapes. Pick one with train.py --preset NAME.
PRESETS: dict[str, dict] = {
    # 31.47M non-embedding, 57.2M total. Sized for ONE 11 hour dual-T4 session:
    # at that compute budget the Chinchilla-optimal model is about 58M total
    # params on roughly 1.2B tokens, and this hits it. Use this if you are doing
    # a single run rather than a multi-week project.
    "1session": {"n_layer": 10, "n_head": 8, "n_embd": 512},
    # 68.83M non-embedding, 101.0M total. Closest clean fit to 67M.
    "67m": {"n_layer": 14, "n_head": 10, "n_embd": 640},
    # 84.95M non-embedding, 123.6M total. The GPT-2 small shape, so your loss
    # curve is directly comparable to published numbers. About 20% slower.
    "125m": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    # 65.44M non-embedding, if you want to be strictly under 67M.
    "65m": {"n_layer": 11, "n_head": 11, "n_embd": 704},
    # 134.51M non-embedding, 173.14M total. Sized for ONE 8.5 hour TPU v3-8
    # session: at roughly 126 effective TFLOP/s that budget is 3.9e18 FLOPs, and
    # the Chinchilla-optimal model there is about 179M total on 3.5B tokens.
    # GPT-2 small's width made deeper, so head_dim stays at 64.
    "tpu1session": {"n_layer": 19, "n_head": 12, "n_embd": 768},
}
PRESETS["gpt2-small"] = PRESETS["125m"]


def non_embedding_params(cfg: GPTConfig) -> int:
    """Exact count for the architecture in model.py: RMSNorm, no biases, 4x GELU MLP.

    Per block: 12*d^2 (attn qkv+out, mlp fc+proj) + 2*d (two norm weights).
    Plus one final norm weight. Token embedding and the tied head are excluded.
    """
    d = cfg.n_embd
    per_block = 12 * d * d + 2 * d
    return cfg.n_layer * per_block + d


def embedding_params(cfg: GPTConfig) -> int:
    n = cfg.vocab_size * cfg.n_embd
    return n if cfg.tie_embeddings else 2 * n


def suggest_configs(target_millions: float = 67.0, top_k: int = 8) -> list[tuple]:
    """Candidate (n_layer, n_embd, n_head, non_emb) sorted by distance to target."""
    target = target_millions * 1e6
    out = []
    for n_embd in range(384, 1025, 64):
        for n_head in (8, 10, 11, 12, 16):
            if n_embd % n_head != 0:
                continue
            if (n_embd // n_head) % 2 != 0 or n_embd // n_head < 32:
                continue
            for n_layer in range(6, 33):
                cfg = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd)
                n = non_embedding_params(cfg)
                # prefer head_dim near 64: it is the sweet spot for T4 tensor cores
                out.append((abs(n - target), abs(n_embd // n_head - 64), n_layer, n_embd, n_head, n))
    out.sort()
    out = [(d, l, e, h, n) for d, _, l, e, h, n in out]
    seen, keep = set(), []
    for _, l, e, h, n in out:
        if (l, e) in seen:
            continue
        seen.add((l, e))
        keep.append((l, e, h, n))
        if len(keep) >= top_k:
            break
    return keep


@dataclass
class TrainConfig:
    """Defaults tuned for 2x T4 with the 14L/640/10H model."""

    micro_batch: int = 8
    grad_accum: int = 8
    block_size: int = 1024

    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 500
    decay_steps: int = 3000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    log_every: int = 20
    eval_every: int = 500
    eval_batches: int = 40
    save_every_min: float = 30.0
    keep_checkpoints: int = 2
    keep_weights: int = 4
    milestone_every_min: float = 90.0

    deadline_hours: float = 11.5
    reserve_minutes: float = 8.0
    seed: int = 1337

    def tokens_per_step(self, world_size: int) -> int:
        return self.micro_batch * self.grad_accum * world_size * self.block_size


if __name__ == "__main__":
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 67.0
    print(f"configs near {target:.0f}M non-embedding params\n")
    print(f"{'layers':>7} {'n_embd':>7} {'heads':>6} {'head_dim':>9} {'non-emb':>10} {'total(tied)':>12}")
    for l, e, h, n in suggest_configs(target, top_k=10):
        cfg = GPTConfig(n_layer=l, n_head=h, n_embd=e)
        total = n + embedding_params(cfg)
        print(f"{l:>7} {e:>7} {h:>6} {e // h:>9} {n / 1e6:>9.2f}M {total / 1e6:>11.2f}M")
    print("\npresets:")
    for name, kw in PRESETS.items():
        c = GPTConfig(**kw)
        print(f"  --preset {name:<11} {c.n_layer}L / {c.n_embd}d / {c.n_head}H -> "
              f"{non_embedding_params(c) / 1e6:.2f}M non-emb, "
              f"{(non_embedding_params(c) + embedding_params(c)) / 1e6:.2f}M total")
