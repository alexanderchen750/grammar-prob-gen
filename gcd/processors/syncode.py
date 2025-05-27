import torch
from pathlib import Path
from .base import LogitsProcessor
from syncode import SyncodeLogitsProcessor, Grammar

class SyncodeProcessor(LogitsProcessor):
    def __init__(self, grammar_path: Path, tokenizer, parse_output_only=True):
        with open(grammar_path, "r") as f:
            grammar = Grammar(f.read())
        self._inner = SyncodeLogitsProcessor(
            grammar=grammar,
            tokenizer=tokenizer,
            parse_output_only=parse_output_only
        )

    def reset(self) -> None:
        self._inner.reset()

    def process(self, prev_ids: torch.LongTensor, logits: torch.Tensor) -> torch.Tensor:
        return self._inner(prev_ids, logits.clone())   # returns (1, vocab)
