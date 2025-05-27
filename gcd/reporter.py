from pathlib import Path
from typing import List
from scorer import StepResult

class Reporter:
    def __init__(self, out_file: Path):
        self.f = open(out_file, "w", encoding="utf-8")

    def write(self, results: List[StepResult], tag: str):
        self.f.write(f"=== {tag} ===\n\n")
        for r in results:
            self.f.write(
                f"Step {r.step} (after {r.prev_token!r})\n"
                f"  → Actual: {r.target_token!r} (P={r.target_prob:.6f})\n"
            )
            for tok, p in zip(r.top_tokens, r.top_probs):
                mark = " ←" if tok == r.target_token else ""
                self.f.write(f"     {tok!r:<15} P={p:.6f}{mark}\n")
            self.f.write("-" * 40 + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
