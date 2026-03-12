import tempfile
from pathlib import Path

import pytest
import torch
import safetensors.torch

from layer_io import LayerLoader, ExpertCache, PinnedBufferPool


@pytest.fixture
def shard_dir(tmp_path: Path) -> Path:
    tensors = {
        "weight_a": torch.randn(64, 64).to(torch.float16),
        "weight_b": torch.randn(64, 64).to(torch.float16),
    }
    safetensors.torch.save_file(tensors, str(tmp_path / "test_shard.safetensors"))
    return tmp_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_layer_loader_load_cpu(shard_dir: Path) -> None:
    loader = LayerLoader(shard_dir, pin_memory=False)
    data = loader.load_shard_cpu("test_shard.safetensors")
    assert "weight_a" in data
    assert "weight_b" in data
    assert data["weight_a"][0].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_layer_loader_load_gpu(shard_dir: Path) -> None:
    loader = LayerLoader(shard_dir)
    data = loader.load_to_gpu("test_shard.safetensors")
    assert data["weight_a"][0].device.type == "cuda"


def test_expert_cache_lru() -> None:
    cache = ExpertCache(max_size=2)
    cache.put("a", {"data": 1})
    cache.put("b", {"data": 2})
    cache.put("c", {"data": 3})
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_pinned_buffer_pool() -> None:
    pool = PinnedBufferPool()
    buf = pool.get((64, 64), torch.float16)
    assert buf.shape == (64, 64)
    pool.release(buf)
    buf2 = pool.get((64, 64), torch.float16)
    assert buf2.shape == (64, 64)
