from typing import Optional, Tuple

import torch

from config import ModelSpec


class KVCache:
    def __init__(
        self,
        spec: ModelSpec,
        max_seq_len: int = 2048,
        gpu_max_seq: int = 1024,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self._spec = spec
        self._max_seq_len = max_seq_len
        self._gpu_max_seq = gpu_max_seq
        self._device = device
        self._dtype = dtype
        self._seq_len = 0

        self._k_cache = []
        self._v_cache = []
        for _ in range(spec.num_layers):
            k = torch.zeros(
                1, spec.num_key_value_heads, max_seq_len, spec.head_dim,
                dtype=dtype,
                device=device if max_seq_len <= gpu_max_seq else "cpu",
                pin_memory=max_seq_len > gpu_max_seq,
            )
            v = torch.zeros(
                1, spec.num_key_value_heads, max_seq_len, spec.head_dim,
                dtype=dtype,
                device=device if max_seq_len <= gpu_max_seq else "cpu",
                pin_memory=max_seq_len > gpu_max_seq,
            )
            self._k_cache.append(k)
            self._v_cache.append(v)

        self._on_gpu = max_seq_len <= gpu_max_seq

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_heads, seq_new, head_dim = k.shape
        start = self._seq_len
        end = start + seq_new

        if self._on_gpu:
            self._k_cache[layer_idx][:, :, start:end, :] = k
            self._v_cache[layer_idx][:, :, start:end, :] = v
            return (
                self._k_cache[layer_idx][:, :, :end, :],
                self._v_cache[layer_idx][:, :, :end, :],
            )
        else:
            k_cpu = k.cpu()
            v_cpu = v.cpu()
            self._k_cache[layer_idx][:, :, start:end, :] = k_cpu
            self._v_cache[layer_idx][:, :, start:end, :] = v_cpu
            return (
                self._k_cache[layer_idx][:, :, :end, :].to(self._device, non_blocking=True),
                self._v_cache[layer_idx][:, :, :end, :].to(self._device, non_blocking=True),
            )

    def advance(self, num_tokens: int) -> None:
        self._seq_len += num_tokens

    def reset(self) -> None:
        self._seq_len = 0
