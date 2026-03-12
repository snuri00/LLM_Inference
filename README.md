# LLM Inference Engine

Layer-by-layer inference engine for running large MoE models on low-VRAM GPUs. Tested with Qwen3-42B-A3B on an RTX 3050 Mobile (4GB VRAM).

Instead of loading the entire model into VRAM, it streams one layer at a time using double-buffered CUDA transfers. For MoE models, only the active experts (selected by the router) are loaded per layer — the rest are skipped entirely.

## How it works

- Model weights are pre-sharded into per-layer safetensors files with NF4 quantization
- Two CUDA streams alternate between compute and PCIe transfer (double buffering)
- Router runs first each layer to identify active experts, then only those are fetched
- KV cache lives on GPU for short sequences, spills to pinned CPU RAM for longer ones
- Dequantized weights are cached in CPU RAM to avoid repeated disk reads

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- bitsandbytes
- safetensors
- transformers (for config parsing and tokenizer)

```
pip install torch bitsandbytes safetensors transformers
```

## Usage

### 1. Download a model

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="DavidAU/Qwen3-42B-A3B-2507-Thinking-Abliterated-uncensored-TOTAL-RECALL-v2-Medium-MASTER-CODER",
    local_dir="./model",
    ignore_patterns=["*.gguf", "*.bin", "*.pt"],
)
```

### 2. Shard the model

```bash
python3 cli.py shard ./model ./shards
```

This splits the HF checkpoint into per-layer NF4 safetensors. Takes a while on the first run. After sharding, you can delete the original model files (keep the tokenizer JSONs).

### 3. Generate

```bash
python3 cli.py generate --shard-dir ./shards --model-path ./model --prompt "Hello, who are you?"
```

### Options

```
--max-seq-len        Maximum sequence length (default: 2048)
--max-new-tokens     Tokens to generate (default: 256)
--temperature        Sampling temperature (default: 0.7)
--top-k              Top-k filtering (default: 50)
--top-p              Nucleus sampling threshold (default: 0.9)
--expert-cache-size  Number of experts kept in GPU memory (default: 64)
```

## Supported models

- Mixtral 8x7B
- Qwen3-MoE (42B-A3B, etc.)
- Qwen2.5-MoE
- DeepSeek-MoE
- Any dense model (layer-by-layer, without MoE optimization)

## VRAM budget (~3.7GB usable on 4GB card)

| Component | Size |
|---|---|
| Double buffer (2 slots) | ~376 MB |
| KV cache (512 tokens) | 64 MB |
| Embedding + LM head | 250 MB |
| Activations + norms | 50 MB |
| CUDA overhead | 300 MB |

## Project structure

```
config.py           Engine config dataclasses
model_adapter.py    HF config parsing, weight name mapping
quantize.py         NF4 quantization via bitsandbytes
shard.py            HF model -> per-layer safetensors
layer_io.py         Selective tensor loading, async GPU transfer, caching
buffers.py          Double-buffer manager with CUDA streams
kv_cache.py         GPU-resident or CPU-offload KV cache
layers/
  attention.py      GQA attention with RoPE and QK-norm
  moe.py            Router-first MoE with batched expert compute
  dense_mlp.py      Dense MLP fallback
  norms.py          RMSNorm
engine.py           Main inference loop
generate.py         Autoregressive token generation
cli.py              Entry point
```

## License

MIT
