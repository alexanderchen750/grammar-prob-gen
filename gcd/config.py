from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-4B"
    grammar_path: Optional[Path] = Path("../grammars/gad.lark")
    device_map: str = "auto"
    dtype: str = "float16"
    top_k: int = 30
    out_dir: Path = Path("outputs")
    corpus: Optional[Path] = None            # None → single-prompt mode
    prompts: List[str] = None                # used when corpus is None

    def make_out_dir(self, tag: str) -> Path:
        d = self.out_dir / tag
        d.mkdir(parents=True, exist_ok=True)
        return d
