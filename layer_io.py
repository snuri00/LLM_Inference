from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import OrderedDict

import torch
from safetensors import safe_open

from quantize import load_quantized_tensor, dequantize_tensor_nf4


class PinnedBufferPool:
    def __init__(self, max_buffers: int = 16):
        self._pool: Dict[Tuple[int, ...], list] = {}
        self._max_buffers = max_buffers

    def get(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        key = (*shape, hash(dtype))
        if key in self._pool and self._pool[key]:
            return self._pool[key].pop()
        buf = torch.empty(shape, dtype=dtype, pin_memory=True)
        return buf

    def release(self, tensor: torch.Tensor) -> None:
        key = (*tensor.shape, hash(tensor.dtype))
        if key not in self._pool:
            self._pool[key] = []
        if len(self._pool[key]) < self._max_buffers:
            self._pool[key].append(tensor)


class CPUShardCache:
    def __init__(self, max_size_mb: float = 20000.0):
        self._cache: OrderedDict[str, Dict[str, Tuple[torch.Tensor, None]]] = OrderedDict()
        self._current_mb: float = 0.0
        self._max_mb = max_size_mb

    def get(self, key: str) -> Optional[Dict[str, Tuple[torch.Tensor, None]]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, data: Dict[str, Tuple[torch.Tensor, None]]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        size_mb = sum(t.numel() * t.element_size() for t, _ in data.values()) / 1024 / 1024
        while self._current_mb + size_mb > self._max_mb and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self._current_mb -= sum(
                t.numel() * t.element_size() for t, _ in evicted.values()
            ) / 1024 / 1024
        self._cache[key] = data
        self._current_mb += size_mb


class LayerLoader:
    def __init__(
        self,
        shard_dir: Path,
        device: str = "cuda",
        pin_memory: bool = True,
        cpu_cache_mb: float = 12000.0,
    ):
        self._shard_dir = shard_dir
        self._device = device
        self._pin_memory = pin_memory
        self._buffer_pool = PinnedBufferPool() if pin_memory else None
        self._cpu_cache = CPUShardCache(max_size_mb=cpu_cache_mb)

    def load_shard_cpu(
        self, shard_name: str
    ) -> Dict[str, Tuple[torch.Tensor, None]]:
        cached = self._cpu_cache.get(shard_name)
        if cached is not None:
            return cached

        path = self._shard_dir / shard_name
        if not path.exists():
            raise FileNotFoundError(f"Shard not found: {path}")
        raw_tensors = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                raw_tensors[key] = f.get_tensor(key)

        result = {}
        seen = set()
        for key in raw_tensors:
            if key in seen or ".__quant_state__." in key:
                continue
            seen.add(key)
            tensor, quant_state = load_quantized_tensor(raw_tensors, key)
            if quant_state is not None:
                gpu_tensor = tensor.cuda()
                quant_state.absmax = quant_state.absmax.cuda()
                tensor = dequantize_tensor_nf4(gpu_tensor, quant_state)
                tensor = tensor.reshape(quant_state.shape).to(torch.float16).cpu()
                del gpu_tensor
                torch.cuda.empty_cache()
            else:
                tensor = tensor.to(torch.float16)
            result[key] = (tensor, None)

        del raw_tensors
        self._cpu_cache.put(shard_name, result)
        return result

    def transfer_to_gpu(
        self,
        cpu_tensors: Dict[str, Tuple[torch.Tensor, None]],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Dict[str, Tuple[torch.Tensor, None]]:
        gpu_tensors = {}
        ctx = torch.cuda.stream(stream) if stream else _nullcontext()
        with ctx:
            for key, (tensor, _) in cpu_tensors.items():
                gpu_tensor = tensor.to(self._device, non_blocking=True)
                gpu_tensors[key] = (gpu_tensor, None)
        return gpu_tensors

    def load_to_gpu(
        self,
        shard_name: str,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Dict[str, Tuple[torch.Tensor, None]]:
        cpu_data = self.load_shard_cpu(shard_name)
        return self.transfer_to_gpu(cpu_data, stream)


class ExpertCache:
    def __init__(self, max_size: int = 64):
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Optional[Dict]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, data: Dict) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = data

    def clear(self) -> None:
        self._cache.clear()


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
