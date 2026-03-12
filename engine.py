from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import torch
import safetensors.torch
from safetensors import safe_open

from config import EngineConfig, ModelSpec
from layer_io import LayerLoader, ExpertCache
from buffers import DoubleBuffer
from kv_cache import KVCache
from layers.norms import RMSNorm
from layers.attention import attention_forward, build_rope_cache
from layers.moe import moe_forward
from layers.dense_mlp import dense_mlp_forward


class InferenceEngine:
    def __init__(self, config: EngineConfig):
        self._config = config
        self._device = config.device
        self._dtype = getattr(torch, config.dtype_compute)

        spec = self._load_spec()
        self._spec = spec
        config.model_spec = spec

        self._loader = LayerLoader(
            config.shard_dir, device=self._device, pin_memory=config.pin_memory
        )
        self._double_buffer = DoubleBuffer(self._loader, device=self._device)
        self._expert_cache = ExpertCache(max_size=config.expert_cache_size)
        self._kv_cache = KVCache(
            spec,
            max_seq_len=config.max_seq_len,
            gpu_max_seq=config.kv_cache_gpu_max_seq,
            device=self._device,
            dtype=self._dtype,
        )

        self._embed_weight = None
        self._lm_head_weight = None
        self._final_norm = RMSNorm(spec.hidden_size, spec.rms_norm_eps).to(self._device)
        self._layer_norms = []

        self._rope_cos = None
        self._rope_sin = None

        self._load_persistent_weights()

    def _load_spec(self) -> ModelSpec:
        spec_path = self._config.shard_dir / "spec.json"
        with open(spec_path) as f:
            d = json.load(f)
        return ModelSpec(**d)

    def _load_persistent_weights(self) -> None:
        shard_dir = self._config.shard_dir

        with safe_open(str(shard_dir / "embed.safetensors"), framework="pt", device=self._device) as f:
            key = list(f.keys())[0]
            self._embed_weight = f.get_tensor(key).to(self._dtype)

        with safe_open(str(shard_dir / "lm_head.safetensors"), framework="pt", device=self._device) as f:
            key = list(f.keys())[0]
            self._lm_head_weight = f.get_tensor(key).to(self._dtype)

        with safe_open(str(shard_dir / "final_norm.safetensors"), framework="pt", device="cpu") as f:
            key = list(f.keys())[0]
            self._final_norm.load_weight(f.get_tensor(key).to(self._dtype))
        self._final_norm = self._final_norm.to(self._device)

        for layer_idx in range(self._spec.num_layers):
            norm_shard = f"layer_{layer_idx:02d}_norm.safetensors"
            norm_path = shard_dir / norm_shard
            norms = {}
            with safe_open(str(norm_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    norms[key] = f.get_tensor(key).to(self._dtype).to(self._device)
            input_norm = RMSNorm(self._spec.hidden_size, self._spec.rms_norm_eps).to(self._device)
            post_norm = RMSNorm(self._spec.hidden_size, self._spec.rms_norm_eps).to(self._device)
            norm_keys = sorted(norms.keys())
            input_norm.load_weight(norms[norm_keys[0]])
            post_norm.load_weight(norms[norm_keys[1]])
            self._layer_norms.append((input_norm, post_norm))

        self._rope_cos, self._rope_sin = build_rope_cache(
            self._config.max_seq_len,
            self._spec.head_dim,
            theta=self._spec.rope_theta,
            device=self._device,
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input_ids, self._embed_weight).to(self._dtype)

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self._final_norm(hidden)
        return torch.nn.functional.linear(hidden, self._lm_head_weight)

    def forward_layer(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        attn_weights: Dict[str, Tuple[torch.Tensor, object]],
    ) -> torch.Tensor:
        input_norm, post_norm = self._layer_norms[layer_idx]

        residual = hidden
        hidden = input_norm(hidden)
        hidden = attention_forward(
            hidden, attn_weights, self._spec, self._kv_cache,
            layer_idx, self._rope_cos, self._rope_sin,
        )
        hidden = residual + hidden

        residual = hidden
        hidden = post_norm(hidden)

        if self._spec.is_moe:
            router_shard = f"layer_{layer_idx:02d}_router.safetensors"
            router_weights = self._loader.load_to_gpu(router_shard)
            hidden = moe_forward(
                hidden, router_weights, self._spec, layer_idx,
                self._loader, self._expert_cache,
                self._double_buffer.transfer_stream,
            )
            del router_weights
        else:
            mlp_shard = f"layer_{layer_idx:02d}_mlp.safetensors"
            mlp_weights = self._loader.load_to_gpu(mlp_shard)
            hidden = dense_mlp_forward(hidden, mlp_weights)
            del mlp_weights

        hidden = residual + hidden
        return hidden

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        num_layers = self._spec.num_layers

        attn_shard_0 = f"layer_00_attn.safetensors"
        attn_data = self._double_buffer.load_initial(attn_shard_0)

        for layer_idx in range(num_layers):
            if layer_idx < num_layers - 1:
                next_shard = f"layer_{layer_idx + 1:02d}_attn.safetensors"
                self._double_buffer.prefetch(next_shard)

            current_attn = self._double_buffer.get_current()
            hidden = self.forward_layer(hidden, layer_idx, current_attn)

            if layer_idx < num_layers - 1:
                self._double_buffer.free_slot(self._double_buffer._current_slot)
                self._double_buffer.wait_and_swap()

        self._double_buffer.free_all()
        self._kv_cache.advance(input_ids.shape[1])
        torch.cuda.empty_cache()

        return self.lm_head(hidden)

    def reset(self) -> None:
        self._kv_cache.reset()
        self._expert_cache.clear()
        self._double_buffer.free_all()
        torch.cuda.empty_cache()
