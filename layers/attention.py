from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import bitsandbytes as bnb

from config import ModelSpec
from kv_cache import KVCache


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    angles = torch.outer(pos, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    seq_len = x.shape[2]
    cos = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def matmul_4bit_proj(
    x: torch.Tensor, weight: torch.Tensor, quant_state: object
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    if quant_state is not None:
        out = bnb.matmul_4bit(x_2d, weight.t(), bias=None, quant_state=quant_state)
    else:
        out = F.linear(x_2d, weight)
    return out.view(*orig_shape[:-1], -1)


def _head_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(orig_dtype)


def attention_forward(
    hidden: torch.Tensor,
    weights: Dict[str, Tuple[torch.Tensor, object]],
    spec: ModelSpec,
    kv_cache: KVCache,
    layer_idx: int,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    bsz, seq_len, _ = hidden.shape

    q_key = [k for k in weights if "q_proj" in k][0]
    k_key = [k for k in weights if "k_proj" in k][0]
    v_key = [k for k in weights if "v_proj" in k][0]
    o_key = [k for k in weights if "o_proj" in k][0]

    q = matmul_4bit_proj(hidden, *weights[q_key])
    k = matmul_4bit_proj(hidden, *weights[k_key])
    v = matmul_4bit_proj(hidden, *weights[v_key])

    q = q.view(bsz, seq_len, spec.num_attention_heads, spec.head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, spec.num_key_value_heads, spec.head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, spec.num_key_value_heads, spec.head_dim).transpose(1, 2)

    q_norm_keys = [key for key in weights if "q_norm" in key]
    k_norm_keys = [key for key in weights if "k_norm" in key]
    if q_norm_keys:
        q_norm_w = weights[q_norm_keys[0]][0]
        k_norm_w = weights[k_norm_keys[0]][0]
        q = _head_rms_norm(q, q_norm_w)
        k = _head_rms_norm(k, k_norm_w)

    offset = kv_cache.seq_len
    q = apply_rope(q, rope_cos, rope_sin, offset)
    k = apply_rope(k, rope_cos, rope_sin, offset)

    k_full, v_full = kv_cache.update(layer_idx, k, v)

    num_kv_groups = spec.num_attention_heads // spec.num_key_value_heads
    if num_kv_groups > 1:
        k_full = k_full.repeat_interleave(num_kv_groups, dim=1)
        v_full = v_full.repeat_interleave(num_kv_groups, dim=1)

    attn_out = F.scaled_dot_product_attention(
        q.to(torch.float16),
        k_full.to(torch.float16),
        v_full.to(torch.float16),
        is_causal=(seq_len > 1),
    )

    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
    out = matmul_4bit_proj(attn_out, *weights[o_key])
    return out
