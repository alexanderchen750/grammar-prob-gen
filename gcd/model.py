from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List

class ModelManager:
    """Load HF model/tokenizer once; provide encode & forward helpers."""
    def __init__(self, model_name: str, device_map="auto", dtype="float16"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=getattr(torch, dtype)
        )
        self.model.eval()

    def encode(self, text: str) -> torch.LongTensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

    @torch.inference_mode()
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.model(input_ids).logits        # (1, seq_len, vocab)
