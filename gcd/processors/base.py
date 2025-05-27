from abc import ABC, abstractmethod
import torch

class LogitsProcessor(ABC):
    """Standard interface every processor must follow."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def process(
        self,
        prev_ids: torch.LongTensor,      # shape (1, i+1)
        logits:  torch.Tensor            # shape (1, vocab)
    ) -> torch.Tensor: ...
