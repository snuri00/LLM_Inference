from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from layers.attention import matmul_4bit_proj


def expert_forward(
    hidden: torch.Tensor,
    weights: Dict[str, Tuple[torch.Tensor, object]],
) -> torch.Tensor:
    gate_key = [k for k in weights if "gate" in k or "w1" in k][0]
    up_key = [k for k in weights if "up" in k or "w3" in k][0]
    down_key = [k for k in weights if "down" in k or "w2" in k][0]

    gate = matmul_4bit_proj(hidden, *weights[gate_key])
    up = matmul_4bit_proj(hidden, *weights[up_key])
    return matmul_4bit_proj(F.silu(gate) * up, *weights[down_key])


def dense_mlp_forward(
    hidden: torch.Tensor,
    weights: Dict[str, Tuple[torch.Tensor, object]],
) -> torch.Tensor:
    return expert_forward(hidden, weights)
