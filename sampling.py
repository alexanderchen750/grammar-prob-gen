#!/usr/bin/env python3
import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# your project imports
from gcd.model import ModelManager
from gcd.processors import SyncodeProcessor
from gcd.tokenScorer import DataCollectingGrammarGuidedLLM


def load_weights(weights_path: str):
    """
    Expect a JSON like:
    {
      "w_state": [...],
      "w_stack": [...],
      "w_rem": float,
      "w_K": float,
      "stack_state2idx": {"S001":0, "S002":1, ...}
    }
    """
    with open(weights_path, "r") as file:
        w = json.load(file)

    w_state = np.array(w["w_state"], dtype=float)
    w_stack = np.array(w["w_stack"], dtype=float)
    w_rem = float(w["w_rem"])
    w_K = float(w["w_K"])
    stack_state2idx: Dict[str, int] = w.get("stack_state2idx", {})

    P_state = len(w_state)
    P_stack = len(w_stack)

    def stack_onehot(names: List[str]) -> np.ndarray:
        # Unknown stack states are ignored (treated as 0)
        oh = np.zeros(P_stack, dtype=float)
        for nm in names or []:
            idx = stack_state2idx.get(nm)
            if idx is not None and 0 <= idx < P_stack:
                oh[idx] += 1.0
        return oh

    def f(s_onehot, stack_names, remainder_text) -> float:
        # Ensure correct shapes
        s_vec = np.asarray(s_onehot, dtype=float)
        if s_vec.shape[0] != P_state:
            # pad or trim to match
            if s_vec.shape[0] < P_state:
                pad = np.zeros(P_state - s_vec.shape[0], dtype=float)
                s_vec = np.concatenate([s_vec, pad])
            else:
                s_vec = s_vec[:P_state]

        v_state = float(s_vec.dot(w_state))
        v_stack = float(stack_onehot(stack_names).dot(w_stack))
        v_rem = w_rem * float(len(remainder_text or ""))

        return v_state + v_stack + v_rem + w_K

    return f


@torch.inference_mode()
def sample_from_ours(
        ggllm: DataCollectingGrammarGuidedLLM,
        syncode_proc: SyncodeProcessor,
        tokenizer,
        model,
        model_mgr,
        f_shift,
        valid_sequences: List[str],
        num_samples: int = 1000,
        max_steps: int = 5,
):
    counts: Dict[str, int] = defaultdict(int)

    # BOS handling: warn if there is no BOS
    if tokenizer.bos_token_id is None:
        print("[WARNING]")
        bos_text = "Generate a sequence of 5 binary digits, provide just the result:"
    else:
        bos_text = tokenizer.bos_token or tokenizer.decode([tokenizer.bos_token_id])

    for _ in range(num_samples):
        # Start from BOS text only if it actually decodes to something (avoid extra spaces)
        sequence = bos_text if bos_text else ""
        ggllm.reset()

        for _t in range(max_steps):
            # Get parser info (with Syncode logits) for the current partial string
            info = ggllm.process_instance_with_syncode(
                text=sequence,
                model_manager=model_mgr,
                syncode_processor=syncode_proc,
            )
            if not info:
                # Could happen for weird partials; just stop this sample
                break

            state = info[-1]

            # Syncode logprobs may be numpy; convert to tensor
            syncode_logprobs = state["syncode_logprobs"]
            if isinstance(syncode_logprobs, (list, tuple, np.ndarray)):
                syncode_logprobs = torch.tensor(syncode_logprobs, device=model.device)
            else:
                syncode_logprobs = torch.as_tensor(syncode_logprobs, device=model.device)
            syncode_logprobs = syncode_logprobs.to(torch.float32)

            # Compute our scalar shift f(...)
            shift = float(
                f_shift(
                    state.get("onehot_current_state", []),
                    state.get("stack", []),
                    state.get("remainder", ""),
                )
            )
            adjusted = syncode_logprobs + shift

            # Sample a token id
            probs = F.softmax(adjusted, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            # Decode and append (strip to avoid leading spaces creeping in)
            next_tok = tokenizer.decode([next_token_id]).strip()
            sequence += next_tok

        # Record validity
        key = sequence if sequence in valid_sequences else f"[INVALID] {sequence}"
        counts[key] += 1

    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description="Sample sequences using 'ours' (Syncode + linear shift).")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--grammar", default="grammars/gad.lark")
    parser.add_argument("--valid-list", default="gadprompts.txt")
    parser.add_argument("--weights",
                        help="Path to saved weights JSON (w_state, w_stack, w_rem, w_K, stack_state2idx).",
                        default="cache_lm_weights/linear_model_weights.json")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--out-json", default="results/ours_samples.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    # Load valid sequences
    with open(args.valid_list, "r") as f:
        valid_sequences = [ln.strip() for ln in f if ln.strip()]

    # Load grammar
    with open(args.grammar, "r") as file:
        grammar_text = file.read()

    # Load model + tokenizer once
    model_mgr = ModelManager(args.model_name)
    model = model_mgr.model
    tokenizer = model_mgr.tokenizer
    model.eval()

    # Syncode processor (constrains logits to grammar)
    syncode_proc = SyncodeProcessor(args.grammar, tokenizer)

    # GGLLM wrapper
    ggllm = DataCollectingGrammarGuidedLLM(
        grammar_text=grammar_text,
        llm_tokenizer_name=args.model_name,
    )

    # Load linear weights and build f
    f_shift = load_weights(args.weights)

    # Sample
    counts = sample_from_ours(
        ggllm=ggllm,
        syncode_proc=syncode_proc,
        tokenizer=tokenizer,
        model=model,
        model_mgr=model_mgr,
        f_shift=f_shift,
        valid_sequences=valid_sequences,
        num_samples=args.num_samples,
        max_steps=args.max_steps,
    )

    # Save
    total = sum(counts.values()) or 1
    probs = {k: v / total for k, v in counts.items()}
    out = {"counts": counts, "probs": probs, "num_samples": total}

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[DONE] Wrote {args.out_json}")


if __name__ == "__main__":
    main()
