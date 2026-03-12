from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VRAMBudget:
    double_buffer_mb: float = 376.0
    kv_cache_mb: float = 64.0
    embedding_mb: float = 250.0
    activations_mb: float = 50.0
    cuda_overhead_mb: float = 300.0

    @property
    def total_mb(self) -> float:
        return (
            self.double_buffer_mb
            + self.kv_cache_mb
            + self.embedding_mb
            + self.activations_mb
            + self.cuda_overhead_mb
        )

    @property
    def single_buffer_mb(self) -> float:
        return self.double_buffer_mb / 2.0


@dataclass
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768
    is_moe: bool = False
    num_experts: int = 1
    num_active_experts: int = 1
    router_jitter_noise: float = 0.0


@dataclass
class EngineConfig:
    model_path: Path
    shard_dir: Path
    device: str = "cuda"
    dtype_compute: str = "float16"
    max_seq_len: int = 2048
    kv_cache_gpu_max_seq: int = 1024
    max_batch_size: int = 1
    pin_memory: bool = True
    num_prefetch_layers: int = 1
    expert_cache_size: int = 64
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 256
    vram_budget: VRAMBudget = field(default_factory=VRAMBudget)
    model_spec: Optional[ModelSpec] = None
