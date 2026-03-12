from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import bitsandbytes as bnb
import safetensors.torch
from bitsandbytes.functional import quantize_nf4, dequantize_nf4


def quantize_tensor_nf4(tensor: torch.Tensor) -> Tuple:
    tensor = tensor.to(torch.float32).contiguous().cuda()
    quantized, quant_state = quantize_nf4(tensor)
    return quantized.cpu(), quant_state


def dequantize_tensor_nf4(
    quantized: torch.Tensor, quant_state: "bnb.functional.QuantState"
) -> torch.Tensor:
    return dequantize_nf4(quantized, quant_state)


def matmul_4bit(
    x: torch.Tensor,
    weight_quantized: torch.Tensor,
    quant_state: "bnb.functional.QuantState",
    bias: torch.Tensor = None,
) -> torch.Tensor:
    return bnb.matmul_4bit(x, weight_quantized.t(), bias=bias, quant_state=quant_state)


def quantize_and_save(
    tensors: Dict[str, torch.Tensor], output_path: Path
) -> None:
    save_dict = {}
    for name, tensor in tensors.items():
        if tensor.ndim < 2 or tensor.shape[0] * tensor.shape[1] < 256:
            save_dict[name] = tensor.to(torch.float16)
            continue
        flat = tensor.to(torch.float32).contiguous().view(-1)
        if flat.numel() % 64 != 0:
            save_dict[name] = tensor.to(torch.float16)
            continue
        quantized, quant_state = quantize_nf4(flat.cuda())
        save_dict[name] = quantized.cpu()
        save_dict[f"{name}.__quant_state__.absmax"] = quant_state.absmax.cpu()
        save_dict[f"{name}.__quant_state__.shape"] = torch.tensor(list(tensor.shape))
        save_dict[f"{name}.__quant_state__.blocksize"] = torch.tensor(
            [quant_state.blocksize]
        )
        save_dict[f"{name}.__quant_state__.dtype"] = torch.tensor(
            [0]
        )
    safetensors.torch.save_file(save_dict, str(output_path))


def load_quantized_tensor(
    tensors: Dict[str, torch.Tensor], name: str
) -> Tuple[torch.Tensor, object]:
    if f"{name}.__quant_state__.absmax" not in tensors:
        return tensors[name], None
    quantized = tensors[name]
    absmax = tensors[f"{name}.__quant_state__.absmax"]
    shape = tuple(tensors[f"{name}.__quant_state__.shape"].tolist())
    blocksize = tensors[f"{name}.__quant_state__.blocksize"].item()
    from bitsandbytes.functional import QuantState
    quant_state = QuantState(
        absmax=absmax,
        shape=shape,
        blocksize=blocksize,
        dtype=torch.float32,
        quant_type="nf4",
    )
    return quantized, quant_state
