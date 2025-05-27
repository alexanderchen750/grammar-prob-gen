from pathlib import Path
from typing import List, Iterable
from config import Config
from model import ModelManager
from processors import NoOpProcessor, SyncodeProcessor
from scorer import TokenProbabilityScorer
from reporter import Reporter
import os
import time

def iter_texts(cfg: Config) -> Iterable[str]:
    if cfg.corpus:
        for line in Path(cfg.corpus).read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield line.strip()
    else:
        yield from cfg.prompts

def run(cfg: Config):
    print(f"Running with model: {cfg.model_name}")
    model_mgr = ModelManager(cfg.model_name, cfg.device_map, cfg.dtype)
    scorer = TokenProbabilityScorer(model_mgr.tokenizer, cfg.top_k)

    print("loading processors")
    processors = {
        "baseline": NoOpProcessor(),
        "syncode":  SyncodeProcessor(cfg.grammar_path, model_mgr.tokenizer),
        # "stackaware": StackAwareProcessor(...),   # drop-in later
    }

    start_total = time.perf_counter()
    for text_idx, text in enumerate(iter_texts(cfg)):

        # Encode
        ids = model_mgr.encode(text)

        # Forward pass
        logits = model_mgr.forward(ids)

        # Processors
        for name, proc in processors.items():
            results = scorer.score(ids, logits, proc)

            # Reporting
            out_dir = cfg.make_out_dir(name)
            rpt = Reporter(out_dir / f"text{str(text_idx).zfill(3)}.txt")
            rpt.write(results, tag=name)
            rpt.close()
    print(f" Total time for data: {time.perf_counter() - start_total:.3f}s")

if __name__ == "__main__":
    with open("sqlprompts.txt", "r", encoding="utf-8") as f:
        prompt_lines = [line.strip() for line in f if line.strip()]

    cfg = Config(
        prompts=prompt_lines
    )
    run(cfg)