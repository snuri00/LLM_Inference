from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoConfig

from config import ModelSpec


MIXTRAL_WEIGHT_MAP = {
    "embed": "model.embed_tokens.weight",
    "lm_head": "lm_head.weight",
    "norm": "model.norm.weight",
    "layer_prefix": "model.layers.{layer}",
    "attn_keys": [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ],
    "norm_keys": [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ],
    "router_key": "block_sparse_moe.gate.weight",
    "expert_keys": [
        "block_sparse_moe.experts.{expert}.w1.weight",
        "block_sparse_moe.experts.{expert}.w2.weight",
        "block_sparse_moe.experts.{expert}.w3.weight",
    ],
}

QWEN_MOE_WEIGHT_MAP = {
    "embed": "model.embed_tokens.weight",
    "lm_head": "lm_head.weight",
    "norm": "model.norm.weight",
    "layer_prefix": "model.layers.{layer}",
    "attn_keys": [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ],
    "norm_keys": [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ],
    "router_key": "mlp.gate.weight",
    "expert_keys": [
        "mlp.experts.{expert}.gate_proj.weight",
        "mlp.experts.{expert}.up_proj.weight",
        "mlp.experts.{expert}.down_proj.weight",
    ],
    "shared_expert_keys": [
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
    ],
}

QWEN3_MOE_WEIGHT_MAP = {
    "embed": "model.embed_tokens.weight",
    "lm_head": "lm_head.weight",
    "norm": "model.norm.weight",
    "layer_prefix": "model.layers.{layer}",
    "attn_keys": [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ],
    "norm_keys": [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ],
    "router_key": "mlp.gate.weight",
    "expert_keys": [
        "mlp.experts.{expert}.gate_proj.weight",
        "mlp.experts.{expert}.up_proj.weight",
        "mlp.experts.{expert}.down_proj.weight",
    ],
}

DENSE_WEIGHT_MAP = {
    "embed": "model.embed_tokens.weight",
    "lm_head": "lm_head.weight",
    "norm": "model.norm.weight",
    "layer_prefix": "model.layers.{layer}",
    "attn_keys": [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ],
    "norm_keys": [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ],
    "mlp_keys": [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ],
}


def detect_model_family(config: AutoConfig) -> str:
    arch = getattr(config, "architectures", [""])[0].lower()
    if "mixtral" in arch:
        return "mixtral"
    if "qwen3moe" in arch:
        return "qwen3_moe"
    if "qwen2moe" in arch:
        return "qwen2_moe"
    if "deepseek" in arch:
        return "deepseek_moe"
    return "dense"


def get_weight_map(family: str) -> Dict:
    if family == "mixtral":
        return MIXTRAL_WEIGHT_MAP
    if family == "qwen3_moe":
        return QWEN3_MOE_WEIGHT_MAP
    if family in ("qwen2_moe", "deepseek_moe"):
        return QWEN_MOE_WEIGHT_MAP
    return DENSE_WEIGHT_MAP


def parse_model_config(model_path: Path) -> Tuple[ModelSpec, Dict]:
    config = AutoConfig.from_pretrained(str(model_path))
    family = detect_model_family(config)
    weight_map = get_weight_map(family)

    is_moe = family != "dense"
    num_experts = getattr(config, "num_local_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "num_experts", 1)
    num_active = getattr(config, "num_experts_per_tok", 1)
    head_dim = getattr(
        config,
        "head_dim",
        config.hidden_size // config.num_attention_heads,
    )

    spec = ModelSpec(
        name=getattr(config, "_name_or_path", model_path.name),
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=getattr(
            config, "num_key_value_heads", config.num_attention_heads
        ),
        head_dim=head_dim,
        vocab_size=config.vocab_size,
        rms_norm_eps=getattr(config, "rms_norm_eps", 1e-5),
        rope_theta=getattr(config, "rope_theta", 10000.0),
        max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
        is_moe=is_moe,
        num_experts=num_experts,
        num_active_experts=num_active,
    )

    return spec, weight_map


def get_layer_weight_names(
    weight_map: Dict, layer_idx: int, num_experts: int = 8
) -> Dict[str, List[str]]:
    prefix = weight_map["layer_prefix"].format(layer=layer_idx)
    result = {}

    result["attn"] = [f"{prefix}.{k}" for k in weight_map["attn_keys"]]
    result["norm"] = [f"{prefix}.{k}" for k in weight_map["norm_keys"]]

    if "router_key" in weight_map:
        result["router"] = [f"{prefix}.{weight_map['router_key']}"]
        for expert_idx in range(num_experts):
            result[f"expert_{expert_idx}"] = [
                f"{prefix}.{k.format(expert=expert_idx)}"
                for k in weight_map["expert_keys"]
            ]
    elif "mlp_keys" in weight_map:
        result["mlp"] = [f"{prefix}.{k}" for k in weight_map["mlp_keys"]]

    return result
