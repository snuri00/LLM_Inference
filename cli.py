import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import EngineConfig, VRAMBudget
from engine import InferenceEngine
from generate import generate
from shard import shard_model


def cmd_shard(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    shard_model(
        model_path,
        output_dir,
        quantize=not args.no_quantize,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    shard_dir = Path(args.shard_dir)
    model_path = Path(args.model_path)

    config = EngineConfig(
        model_path=model_path,
        shard_dir=shard_dir,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        expert_cache_size=args.expert_cache_size,
    )

    print("Loading engine...")
    engine = InferenceEngine(config)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    eos_id = tokenizer.eos_token_id

    prompt = args.prompt
    if prompt is None:
        prompt = input("Prompt: ")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"\n{prompt}", end="", flush=True)

    def on_token(token_id: int) -> None:
        text = tokenizer.decode([token_id], skip_special_tokens=True)
        print(text, end="", flush=True)

    t0 = time.perf_counter()
    tokens = generate(
        engine,
        input_ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        eos_token_id=eos_id,
        stream_callback=on_token,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n\n--- {len(tokens)} tokens in {elapsed:.1f}s ({len(tokens)/elapsed:.1f} tok/s) ---")
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak VRAM: {peak_mb:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-VRAM MoE Inference Engine")
    subparsers = parser.add_subparsers(dest="command")

    sp_shard = subparsers.add_parser("shard", help="Shard HF model into per-layer safetensors")
    sp_shard.add_argument("model_path", type=str)
    sp_shard.add_argument("output_dir", type=str)
    sp_shard.add_argument("--no-quantize", action="store_true")

    sp_gen = subparsers.add_parser("generate", help="Generate text")
    sp_gen.add_argument("--shard-dir", type=str, required=True)
    sp_gen.add_argument("--model-path", type=str, required=True)
    sp_gen.add_argument("--prompt", type=str, default=None)
    sp_gen.add_argument("--max-seq-len", type=int, default=2048)
    sp_gen.add_argument("--max-new-tokens", type=int, default=256)
    sp_gen.add_argument("--temperature", type=float, default=0.7)
    sp_gen.add_argument("--top-k", type=int, default=50)
    sp_gen.add_argument("--top-p", type=float, default=0.9)
    sp_gen.add_argument("--expert-cache-size", type=int, default=4)

    args = parser.parse_args()
    if args.command == "shard":
        cmd_shard(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
