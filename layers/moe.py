from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import ModelSpec
from layer_io import LayerLoader, ExpertCache
from layers.dense_mlp import expert_forward


def route(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    num_active: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    bsz_seq = hidden.view(-1, hidden.shape[-1])
    logits = F.linear(bsz_seq.float(), router_weight.float())
    probs = F.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, k=num_active, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    active_ids = indices.unique().tolist()
    return weights, indices, active_ids


def moe_forward(
    hidden: torch.Tensor,
    router_weights: Dict[str, Tuple[torch.Tensor, object]],
    spec: ModelSpec,
    layer_idx: int,
    loader: LayerLoader,
    expert_cache: ExpertCache,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    router_key = [k for k in router_weights if "gate" in k or "router" in k][0]
    router_w = router_weights[router_key][0]
    if router_weights[router_key][1] is not None:
        from bitsandbytes.functional import dequantize_nf4
        router_w = dequantize_nf4(router_w, router_weights[router_key][1])

    weights, indices, active_ids = route(hidden, router_w, spec.num_active_experts)

    expert_data = {}
    for eid in active_ids:
        cache_key = f"layer_{layer_idx:02d}_expert_{eid}"
        cached = expert_cache.get(cache_key)
        if cached is not None:
            expert_data[eid] = cached
        else:
            shard_name = f"layer_{layer_idx:02d}_expert_{eid}.safetensors"
            data = loader.load_to_gpu(shard_name, stream)
            if stream is not None:
                stream.synchronize()
            expert_data[eid] = data
            expert_cache.put(cache_key, data)

    orig_shape = hidden.shape
    flat_hidden = hidden.view(-1, orig_shape[-1])
    output = torch.zeros_like(flat_hidden)

    for eid in active_ids:
        mask = (indices == eid).any(dim=-1)
        if not mask.any():
            continue
        token_indices = mask.nonzero(as_tuple=True)[0]
        expert_input = flat_hidden[token_indices]
        k_mask = indices[token_indices] == eid
        expert_weights = (weights[token_indices] * k_mask.float()).sum(dim=-1, keepdim=True)
        expert_out = expert_forward(expert_input, expert_data[eid])
        output[token_indices] += expert_weights * expert_out

    return output.view(orig_shape)
