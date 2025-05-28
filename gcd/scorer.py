import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

@dataclass
class StepResult:
    step: int
    prev_token: str
    target_token: str
    target_prob: float
    top_tokens: List[str]
    top_probs: List[float]

class TokenProbabilityScorer:
    def __init__(self, tokenizer, top_k: int = 10):
        self.tok = tokenizer
        self.k = top_k

    def score(
        self,
        ids: torch.LongTensor,           # (1, seq_len)
        logits_seq: torch.Tensor,        # (1, seq_len, vocab)
        processor
    ) -> List[StepResult]:

        processor.reset()
        results: List[StepResult] = []

        for i in range(-1, ids.size(1) - 1):
            prev_ids = ids[:, : i + 1] if i >= 0 else ids[:, :0]  # Empty prefix for i == -1
            logits_i = logits_seq[0, i] if i >= 0 else logits_seq[0, 0]  # Use first logits for i == -1 # (vocab,)
            proc_logits = processor.process(prev_ids, logits_i.unsqueeze(0))[0]

            log_probs = F.log_softmax(proc_logits, dim=-1)
            target_id = ids[0, i + 1].item()

            topk = torch.topk(log_probs, k=self.k)

            results.append(
                StepResult(
                    step=i + 1,
                    prev_token=self.tok.decode([ids[0, i]]) if i >= 0 else "<BOS>",
                    target_token=self.tok.decode([target_id]),
                    target_prob=log_probs[target_id].item(),
                    top_tokens=[self.tok.decode([t.item()]) for t in topk.indices],
                    top_probs=topk.values.tolist(),
                )
            )
        return results
