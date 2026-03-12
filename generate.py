from typing import List, Optional

import torch
import torch.nn.functional as F

from config import EngineConfig
from engine import InferenceEngine


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> int:
    logits = logits[:, -1, :].float()

    if temperature <= 0:
        return logits.argmax(dim=-1).item()

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        values, _ = torch.topk(logits, top_k, dim=-1)
        logits[logits < values[:, -1:]] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return token.item()


@torch.inference_mode()
def generate(
    engine: InferenceEngine,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
    stream_callback=None,
) -> List[int]:
    engine.reset()
    device = engine._device
    input_ids = input_ids.to(device)

    logits = engine.forward(input_ids)
    next_token = sample_token(logits, temperature, top_k, top_p)
    generated = [next_token]

    if stream_callback:
        stream_callback(next_token)

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and next_token == eos_token_id:
            break

        token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits = engine.forward(token_tensor)
        next_token = sample_token(logits, temperature, top_k, top_p)
        generated.append(next_token)

        if stream_callback:
            stream_callback(next_token)

    return generated
