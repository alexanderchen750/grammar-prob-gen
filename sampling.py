import os
import re
import gc
import json
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from gcd.processors import SyncodeProcessor
from pathlib import Path


number_of_samples = 1000
batch_size = 100
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)


df = pd.read_pickle("training_data/grammar_data_df.pkl")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
prompt = "Generate a sequence of 5 binary digits following the format: either exactly 00000, or a 1 followed by any 4 binary digits. Provide just the result:"

with open("gadprompts.txt", 'r') as file:
    valid_sequences = [line.strip() for line in file if line.strip()]


df["sequence_id"] = (df.index // 5).astype(int)

# parser-state dimension
P_state = len(df["parser_state_onehot"].iloc[0])

# stack one-hot builder
sample_stack = df["stack"].iloc[0]
if all(isinstance(e, str) for e in sample_stack):
    unique_states = sorted({name for lst in df["stack"] for name in lst})
    state2idx = {name: i for i, name in enumerate(unique_states)}
    P_stack = len(unique_states)

    def stack_onehot(names):
        oh = np.zeros(P_stack, dtype=float)
        for nm in names:
            oh[state2idx[nm]] += 1.0
        return oh
else:
    raise ValueError(f"wrong stack format: {type(sample_stack)}")

def next_id_or_argmax(row_df):
    val = row_df["next_token"]
    if val is not None and not pd.isna(val):
        ids = tokenizer.encode(val, add_special_tokens=False)
        return ids[0] if ids else tokenizer.eos_token_id
    else:
        logprobs = np.array(row_df["syncode_logprobs"])
        return int(np.argmax(logprobs))

df["next_token_id"] = df.apply(next_id_or_argmax, axis=1)


X_rows, y_rows = [], []

for seq_id, group in df.groupby("sequence_id"):
    sequence_diffs = []
    for _, row in group.iterrows():
        syncode = np.array(row["syncode_logprobs"])
        base = np.array(row["baseline_logprobs"])
        sequence_diffs.append(base - syncode)
    sequence_diff = np.stack(sequence_diffs).sum(axis=0)

    for _, row in group.iterrows():
        state_vec = np.asarray(row["parser_state_onehot"], dtype=float)
        stack_vec = stack_onehot(row["stack"])
        rem_len = len(row["remainder"])
        x = np.concatenate([state_vec, stack_vec, [rem_len]])
        X_rows.append(x)
        next_id = row["next_token_id"]
        y_rows.append(sequence_diff[next_id])

X = np.vstack(X_rows)
y = np.array(y_rows)

lin = LinearRegression()
lin.fit(X, y)
w = lin.coef_
w_state = w[:P_state]
w_stack = w[P_state:P_state + P_stack]
w_rem = w[-1]
w_K = lin.intercept_

def f_shift_scalar(s_onehot, stack_idxs, rem):
    v_state = np.asarray(s_onehot, dtype=float).dot(w_state)
    v_stack = stack_onehot(stack_idxs).dot(w_stack)
    v_rem = w_rem * len(rem)
    return float(v_state + v_stack + v_rem + w_K)

# ----------------------------
# Our logits processor
# ----------------------------

from parserState import ParserStateExtractor

class BatchSyncodeCallable:
    """
    Adapt per-row SyncodeProcessor.process(prev_ids, logits) -> logits
    to a batch callable: (input_ids, scores) -> scores
    """
    def __init__(self, syncode_processor):
        self.proc = syncode_processor

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Expect input_ids: (B, L), scores: (B, V)
        B = input_ids.size(0)

        # Keep originals to restore dtype/device at the end
        orig_dtype = scores.dtype
        orig_device = scores.device

        # Work device: use scores' device by default
        work_device = orig_device

        outs = []
        for b in range(B):
            prev_ids_b = input_ids[b:b+1].to(device=work_device, dtype=torch.long, non_blocking=True)  # (1, L)
            # Many custom processors assume float32 logits
            scores_b = scores[b:b+1].to(device=work_device, dtype=torch.float32, non_blocking=True)    # (1, V)

            out_b = self.proc.process(prev_ids_b, scores_b)  # should return (1, V) float32
            # Be defensive about dtype/device coming back
            out_b = out_b.to(device=work_device, dtype=torch.float32, non_blocking=True)

            outs.append(out_b)

        out = torch.cat(outs, dim=0)  # (B, V)
        # Cast back to the dtype HF generation is using (e.g., float16 on GPU)
        out = out.to(device=orig_device, dtype=orig_dtype, non_blocking=True)
        return out

class SyncodeWithFProcessor(LogitsProcessor):
    """
    1) (Optionally) apply Syncode logits masking (callable you pass in).
    2) For each non -inf token, clone a ParserStateExtractor positioned at the current prefix,
       advance by that token, extract (state_onehot, stack, remainder),
       evaluate f_shift, and add it to that token's logit.
    """

    def __init__(
        self,
        *,
        grammar_text: str,
        tokenizer,
        parser_state_extractor_cls,
        f_shift_fn,                            # callable(state_onehot, stack, remainder) -> float
        syncode_proc=None,                     # optional: callable(input_ids, scores) -> scores
        stack_context_length: int = 3,
        decode_token_fn=None
    ):
        self.grammar_text = grammar_text
        self.tokenizer = tokenizer
        self.ParserStateExtractorCls = parser_state_extractor_cls
        self.f_shift_fn = f_shift_fn
        self.syncode_proc = syncode_proc       # if None, no grammar masking
        self.stack_context_length = stack_context_length
        self.decode_token_fn = decode_token_fn or (lambda tid: tokenizer.decode([tid], skip_special_tokens=True))

    def _build_extractor_for_prefix(self, prefix_text: str):
        ex = self.ParserStateExtractorCls(self.grammar_text)
        # position the extractor at the prefix
        ex.advance_parser(prefix_text, top_k=self.stack_context_length, prefix_text=prefix_text)
        return ex

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # 1) (optional) apply Syncode logits processor first
        if self.syncode_proc is not None:
            scores = self.syncode_proc(input_ids, scores)

        B, V = scores.shape
        neginf_mask = torch.isneginf(scores)

        for b in range(B):
            prefix_text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            base_extractor = self._build_extractor_for_prefix(prefix_text)

            valid_ids = torch.nonzero(~neginf_mask[b], as_tuple=False).flatten().tolist()
            if not valid_ids:
                continue

            for tid in valid_ids:
                tok_text = self.decode_token_fn(tid)

                # clone + advance by single token
                ex_next = copy.deepcopy(base_extractor)
                res = ex_next.advance_parser(tok_text, top_k=self.stack_context_length,
                                             prefix_text=prefix_text + tok_text)

                state_oh = res.get('onehot_current_state', [])
                stack_names = res.get('stack', [])
                remainder = getattr(ex_next, "current_remainder", "")

                delta = float(self.f_shift_fn(state_oh, stack_names, remainder))
                if delta != 0.0:
                    scores[b, tid] = scores[b, tid] + scores.new_tensor(delta)

        # Never revive disallowed tokens
        scores = torch.where(neginf_mask, scores.new_full(scores.shape, float("-inf")), scores)
        return scores

# ----------------------------
# Build model and logits processor
# ----------------------------
with open("grammars/gad.lark", 'r') as file:
    grammar_text = file.read()

syncode_processor = SyncodeProcessor(
    grammar_path=Path("grammars/gad.lark"),
    tokenizer=tokenizer,
    parse_output_only=True,
)

syncode_proc = BatchSyncodeCallable(syncode_processor)

processor = SyncodeWithFProcessor(
    grammar_text=grammar_text,
    tokenizer=tokenizer,
    parser_state_extractor_cls=ParserStateExtractor,
    f_shift_fn=f_shift_scalar,
    syncode_proc=syncode_proc,
    stack_context_length=3
)

# NOTE: Model can be large; ensure enough VRAM or switch to quantized weights if needed.
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", device_map="auto", torch_dtype=getattr(torch, "float16")
)
model.eval()

# ----------------------------
# Sampling via our logits processor
# ----------------------------
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
processors = LogitsProcessorList([processor])

our_counts = {}
num_batches = (number_of_samples + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    curr_batch_size = min(batch_size, number_of_samples - batch_idx * batch_size)
    try:
        outputs = model.generate(
            input_ids=inputs["input_ids"].repeat(curr_batch_size, 1),
            max_new_tokens=20,                 # generous, we'll parse out the 5 bits
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=processors
        )
    except Exception as e:
        print("Generation error:", e)
        outputs = []

    for seq_tensor in outputs:
        text = tokenizer.decode(seq_tensor, skip_special_tokens=True)
        raw_response = text.replace(prompt, "").strip()
        raw_response = re.sub(r"\s+", " ", raw_response)

        m = re.search(r"\b[01]{5}\b", raw_response)
        if m:
            candidate = m.group(0)
            key = candidate if candidate in valid_sequences else f"[INVALID] {raw_response}"
        else:
            key = f"[INVALID] {raw_response}"

        our_counts[key] = our_counts.get(key, 0) + 1

# ----------------------------
# Build distributions & plots (ours only)
# ----------------------------
total = sum(our_counts.values()) if our_counts else 1
p_ours_all = {seq: cnt / total for seq, cnt in our_counts.items()}
p_ours = {seq: prob for seq, prob in p_ours_all.items() if not seq.startswith("[INVALID]")}
p_ours_invalid = {seq: prob for seq, prob in p_ours_all.items() if seq.startswith("[INVALID]")}

print("\nOurs raw counts:")
for seq in valid_sequences:
    print(f"{seq}: {our_counts.get(seq, 0)}")

print("\nNormalized Our distribution:")
for seq, prob in sorted(p_ours.items()):
    print(f"{seq}: {prob:.6f}")

sorted_seqs = sorted(valid_sequences)
x = np.arange(len(sorted_seqs))

def plot_distribution(probs, method_name, filename):
    plt.figure(figsize=(15, 5))
    plt.bar(x, [probs.get(seq, 0.0) for seq in sorted_seqs], width=0.6)
    plt.xticks(x, sorted_seqs, rotation=90)
    plt.xlabel("Sequences")
    plt.ylabel("Probability")
    plt.title(f"{method_name} Distribution over Accepted Sequences")
    plt.tight_layout()
    plt.savefig(f"plots/{filename}")
    plt.close()

plot_distribution(p_ours, "Ours (LogitsProcessor)", "ours_lp_distribution.png")

invalid_keys = sorted(p_ours_invalid.keys())
if invalid_keys:
    x_invalid = np.arange(len(invalid_keys))
    def plot_invalid_distribution(probs, method_name, filename):
        plt.figure(figsize=(15, 7), constrained_layout=True)
        plt.bar(x_invalid, [probs.get(seq, 0.0) for seq in invalid_keys], width=0.6)
        plt.xticks(x_invalid, invalid_keys, rotation=90, ha='left')
        plt.xlabel("Invalid Sequences")
        plt.ylabel("Probability")
        plt.title(f"{method_name} Distribution over Invalid Sequences")
        plt.savefig(f"plots/{filename}")
        plt.close()
    plot_invalid_distribution(p_ours_invalid, "Ours (LogitsProcessor)", "ours_lp_invalid_distribution.png")

# ----------------------------
# Save results JSON (ours only)
# ----------------------------
output_data = {
    "valid": {
        "p_ours": p_ours
    },
    "invalid": {
        "p_ours_invalid": p_ours_invalid
    }
}
with open("results/our_distribution_data.json", "w") as f:
    json.dump(output_data, f, indent=2)

# Cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
