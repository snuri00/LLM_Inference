import json
from pathlib import Path

import pytest
import torch

from layers.norms import RMSNorm
from layers.attention import build_rope_cache, apply_rope
from layers.dense_mlp import expert_forward


def test_rmsnorm_shape() -> None:
    norm = RMSNorm(64)
    x = torch.randn(1, 10, 64)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_normalized() -> None:
    norm = RMSNorm(64)
    x = torch.randn(1, 10, 64) * 100
    out = norm(x)
    rms = out.float().pow(2).mean(-1).sqrt()
    assert rms.max() < 10.0


def test_rope_cache_shape() -> None:
    cos, sin = build_rope_cache(128, 16, device="cpu")
    assert cos.shape == (128, 8)
    assert sin.shape == (128, 8)


def test_apply_rope_shape() -> None:
    cos, sin = build_rope_cache(128, 16, device="cpu")
    x = torch.randn(1, 4, 10, 16)
    out = apply_rope(x, cos, sin, offset=0)
    assert out.shape == x.shape


def test_dense_mlp_forward() -> None:
    hidden = 64
    intermediate = 128
    weights = {
        "gate_proj.weight": (torch.randn(intermediate, hidden), None),
        "up_proj.weight": (torch.randn(intermediate, hidden), None),
        "down_proj.weight": (torch.randn(hidden, intermediate), None),
    }
    x = torch.randn(1, 10, hidden)
    out = expert_forward(x, weights)
    assert out.shape == (1, 10, hidden)
