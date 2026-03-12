from typing import Dict, Optional, Tuple

import torch

from layer_io import LayerLoader


class DoubleBuffer:
    def __init__(self, loader: LayerLoader, device: str = "cuda"):
        self._loader = loader
        self._device = device
        self._compute_stream = torch.cuda.Stream(device=device)
        self._transfer_stream = torch.cuda.Stream(device=device)
        self._slots: list = [None, None]
        self._events: list = [
            torch.cuda.Event(enable_timing=False),
            torch.cuda.Event(enable_timing=False),
        ]
        self._current_slot = 0

    @property
    def compute_stream(self) -> torch.cuda.Stream:
        return self._compute_stream

    @property
    def transfer_stream(self) -> torch.cuda.Stream:
        return self._transfer_stream

    def prefetch(self, shard_name: str) -> None:
        next_slot = 1 - self._current_slot
        gpu_data = self._loader.load_to_gpu(shard_name, self._transfer_stream)
        self._slots[next_slot] = gpu_data
        self._transfer_stream.record_event(self._events[next_slot])

    def get_current(self) -> Optional[Dict[str, Tuple[torch.Tensor, object]]]:
        return self._slots[self._current_slot]

    def wait_and_swap(self) -> Dict[str, Tuple[torch.Tensor, object]]:
        next_slot = 1 - self._current_slot
        self._compute_stream.wait_event(self._events[next_slot])
        self._current_slot = next_slot
        return self._slots[self._current_slot]

    def load_initial(self, shard_name: str) -> Dict[str, Tuple[torch.Tensor, object]]:
        gpu_data = self._loader.load_to_gpu(shard_name, self._transfer_stream)
        self._transfer_stream.synchronize()
        self._slots[self._current_slot] = gpu_data
        return gpu_data

    def free_slot(self, slot: Optional[int] = None) -> None:
        if slot is None:
            slot = self._current_slot
        if self._slots[slot] is not None:
            self._slots[slot] = None
            torch.cuda.empty_cache()

    def free_all(self) -> None:
        self.free_slot(0)
        self.free_slot(1)
