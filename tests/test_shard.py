import json
import tempfile
from pathlib import Path

import pytest
import torch
import safetensors.torch

from config import ModelSpec


def create_fake_model(tmp_path: Path, num_layers: int = 2, num_experts: int = 2) -> Path:
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()

    hidden = 64
    intermediate = 128
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden // num_heads
    vocab = 256

    tensors = {}
    tensors["model.embed_tokens.weight"] = torch.randn(vocab, hidden)
    tensors["lm_head.weight"] = torch.randn(vocab, hidden)
    tensors["model.norm.weight"] = torch.randn(hidden)

    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(num_kv_heads * head_dim, hidden)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(num_kv_heads * head_dim, hidden)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden)
        tensors[f"{prefix}.block_sparse_moe.gate.weight"] = torch.randn(num_experts, hidden)
        for expert in range(num_experts):
            tensors[f"{prefix}.block_sparse_moe.experts.{expert}.w1.weight"] = torch.randn(intermediate, hidden)
            tensors[f"{prefix}.block_sparse_moe.experts.{expert}.w2.weight"] = torch.randn(hidden, intermediate)
            tensors[f"{prefix}.block_sparse_moe.experts.{expert}.w3.weight"] = torch.randn(intermediate, hidden)

    safetensors.torch.save_file(tensors, str(model_dir / "model.safetensors"))

    weight_map = {k: "model.safetensors" for k in tensors}
    index = {"metadata": {}, "weight_map": weight_map}
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    config = {
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "num_hidden_layers": num_layers,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 512,
        "num_local_experts": num_experts,
        "num_experts_per_tok": 2,
        "model_type": "mixtral",
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    return model_dir


@pytest.fixture
def fake_model(tmp_path: Path) -> Path:
    return create_fake_model(tmp_path)


def test_shard_creates_expected_files(fake_model: Path, tmp_path: Path) -> None:
    from shard import shard_model

    output_dir = tmp_path / "shards"
    shard_model(fake_model, output_dir, quantize=False)

    assert (output_dir / "spec.json").exists()
    assert (output_dir / "embed.safetensors").exists()
    assert (output_dir / "lm_head.safetensors").exists()
    assert (output_dir / "final_norm.safetensors").exists()
    assert (output_dir / "layer_00_attn.safetensors").exists()
    assert (output_dir / "layer_00_norm.safetensors").exists()
    assert (output_dir / "layer_00_router.safetensors").exists()
    assert (output_dir / "layer_00_expert_0.safetensors").exists()
    assert (output_dir / "layer_00_expert_1.safetensors").exists()
    assert (output_dir / "layer_01_attn.safetensors").exists()


def test_spec_json_correct(fake_model: Path, tmp_path: Path) -> None:
    from shard import shard_model

    output_dir = tmp_path / "shards"
    shard_model(fake_model, output_dir, quantize=False)

    with open(output_dir / "spec.json") as f:
        spec = json.load(f)

    assert spec["num_layers"] == 2
    assert spec["hidden_size"] == 64
    assert spec["num_experts"] == 2
    assert spec["is_moe"] is True
