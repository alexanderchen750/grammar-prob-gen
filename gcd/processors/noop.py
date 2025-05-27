import torch
from .base import LogitsProcessor

class NoOpProcessor(LogitsProcessor):
    """Baseline: return logits unchanged."""

    def reset(self) -> None:
        pass

    def process(self, prev_ids: torch.LongTensor, logits: torch.Tensor) -> torch.Tensor:
        return logits
