import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import safetensors.torch
from safetensors import safe_open

from config import ModelSpec
from model_adapter import parse_model_config, get_layer_weight_names


def load_safetensors_index(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        return index["weight_map"]
    st_files = list(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")
    weight_map = {}
    for st_file in st_files:
        with safe_open(str(st_file), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = st_file.name
    return weight_map


def group_weights_by_file(
    weight_names: List[str], weight_map: Dict[str, str]
) -> Dict[str, List[str]]:
    file_groups: Dict[str, List[str]] = {}
    for name in weight_names:
        if name not in weight_map:
            continue
        filename = weight_map[name]
        if filename not in file_groups:
            file_groups[filename] = []
        file_groups[filename].append(name)
    return file_groups


def load_tensors_from_files(
    model_path: Path,
    file_groups: Dict[str, List[str]],
) -> Dict[str, torch.Tensor]:
    tensors = {}
    for filename, names in file_groups.items():
        filepath = model_path / filename
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
    return tensors


def strip_prefix(tensors: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in tensors.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            if new_key.startswith("."):
                new_key = new_key[1:]
            result[new_key] = value
        else:
            result[key] = value
    return result


def shard_model(
    model_path: Path,
    output_dir: Path,
    quantize: bool = True,
    device: str = "cuda",
) -> None:
    spec, weight_map_template = parse_model_config(model_path)
    hf_weight_map = load_safetensors_index(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    embed_name = weight_map_template["embed"]
    lm_head_name = weight_map_template["lm_head"]
    final_norm_name = weight_map_template["norm"]

    global_names = [embed_name, lm_head_name, final_norm_name]
    file_groups = group_weights_by_file(global_names, hf_weight_map)
    global_tensors = load_tensors_from_files(model_path, file_groups)

    embed_tensors = {embed_name: global_tensors[embed_name]}
    safetensors.torch.save_file(
        {k: v.to(torch.float16) for k, v in embed_tensors.items()},
        str(output_dir / "embed.safetensors"),
    )

    head_tensors = {}
    if lm_head_name in global_tensors:
        head_tensors[lm_head_name] = global_tensors[lm_head_name]
    else:
        head_tensors[lm_head_name] = global_tensors[embed_name]
    safetensors.torch.save_file(
        {k: v.to(torch.float16) for k, v in head_tensors.items()},
        str(output_dir / "lm_head.safetensors"),
    )

    safetensors.torch.save_file(
        {final_norm_name: global_tensors[final_norm_name].to(torch.float16)},
        str(output_dir / "final_norm.safetensors"),
    )

    del global_tensors
    torch.cuda.empty_cache()

    for layer_idx in range(spec.num_layers):
        print(f"Sharding layer {layer_idx}/{spec.num_layers - 1}")
        layer_names = get_layer_weight_names(
            weight_map_template, layer_idx, spec.num_experts
        )
        prefix = weight_map_template["layer_prefix"].format(layer=layer_idx)

        for group_name, names in layer_names.items():
            file_groups = group_weights_by_file(names, hf_weight_map)
            tensors = load_tensors_from_files(model_path, file_groups)
            tensors = strip_prefix(tensors, prefix)

            out_name = f"layer_{layer_idx:02d}_{group_name}.safetensors"
            out_path = output_dir / out_name

            if quantize and group_name not in ("norm", "router"):
                from quantize import quantize_and_save
                quantize_and_save(tensors, out_path)
            else:
                safetensors.torch.save_file(
                    {k: v.to(torch.float16) for k, v in tensors.items()},
                    str(out_path),
                )
            del tensors
            torch.cuda.empty_cache()

    spec_dict = {
        "name": spec.name,
        "num_layers": spec.num_layers,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_key_value_heads": spec.num_key_value_heads,
        "head_dim": spec.head_dim,
        "vocab_size": spec.vocab_size,
        "rms_norm_eps": spec.rms_norm_eps,
        "rope_theta": spec.rope_theta,
        "max_position_embeddings": spec.max_position_embeddings,
        "is_moe": spec.is_moe,
        "num_experts": spec.num_experts,
        "num_active_experts": spec.num_active_experts,
    }
    with open(output_dir / "spec.json", "w") as f:
        json.dump(spec_dict, f, indent=2)

    print(f"Sharding complete: {output_dir}")
