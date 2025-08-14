import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from syncode import Syncode
import matplotlib.pyplot as plt
import os
import re
import json
import gc
from lark import Lark
from gcd.processors import SyncodeProcessor, BatchSyncodeCallable, SyncodeWithFProcessor
from pathlib import Path
from parserState.ParserStateExtractor import ParserStateExtractor

MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0


def check_validity_of_output(parser, output):
    try:
        parser.parse(output)
        return True
    except Exception as e:
        return False


with open("grammars/bv4.lark", 'r') as file:
    grammar_text = file.read()

parser = Lark(grammar_text, parser='lalr', lexer='contextual')

number_of_samples = 100
os.makedirs("plots", exist_ok=True)

df = pd.read_pickle("training_data/grammar_data_df.pkl")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

syncode_processor = SyncodeProcessor(
    grammar_path=Path("grammars/bv4.lark"),
    tokenizer=tokenizer,
    parse_output_only=True,
)

syncode_proc = BatchSyncodeCallable(syncode_processor)

prompt = """You are an expert in program synthesis. You are tasked with solving a Syntax-Guided Synthesis (SyGuS) problem. Your goal is to output a function that should produce outputs that satisfy a series of constraints when given specific inputs.

Question:
(set-logic BV)

(synth-fun inv ((s (BitVec 4)) (t (BitVec 4))) (BitVec 4)
    ((Start (BitVec 4)))
    ((Start (BitVec 4) (s t #x0 #x8 #x7 (bvneg Start) (bvnot Start) (bvadd Start Start) (bvsub Start Start) (bvand Start Start) (bvlshr Start Start) (bvor Start Start) (bvshl Start Start)))))

(declare-var s (BitVec 4))
(declare-var t (BitVec 4))
(define-fun udivtotal ((a (BitVec 4)) (b (BitVec 4))) (BitVec 4)
    (ite (= b #x0) #xF (bvudiv a b)))
(define-fun uremtotal ((a (BitVec 4)) (b (BitVec 4))) (BitVec 4)
    (ite (= b #x0) a (bvurem a b)))
(define-fun min () (BitVec 4)
    (bvnot (bvlshr (bvnot #x0) #x1)))
(define-fun max () (BitVec 4)
    (bvnot min))
(define-fun l ((s (BitVec 4)) (t (BitVec 4))) Bool
    (bvsle (bvnot (inv s t)) t))
(define-fun SC ((s (BitVec 4)) (t (BitVec 4))) Bool
    true)
(constraint (=> (SC s t) (l s t)))

(check-synth)
Solution:
(define-fun inv ((s (BitVec 4)) (t (BitVec 4))) (BitVec 4) #b0111)

Question:
(set-logic BV)

(synth-fun inv ((s (BitVec 4)) (t (BitVec 4))) (BitVec 4)
    ((Start (BitVec 4)))
    ((Start (BitVec 4) (s t #x0 #x8 #x7 (bvneg Start) (bvnot Start) (bvadd Start Start) (bvsub Start Start) (bvand Start Start) (bvlshr Start Start) (bvor Start Start) (bvshl Start Start)))))

(declare-var s (BitVec 4))
(declare-var t (BitVec 4))
(define-fun udivtotal ((a (BitVec 4)) (b (BitVec 4))) (BitVec 4)
    (ite (= b #x0) #xF (bvudiv a b)))
(define-fun uremtotal ((a (BitVec 4)) (b (BitVec 4))) (BitVec 4)
    (ite (= b #x0) a (bvurem a b)))
(define-fun min () (BitVec 4)
    (bvnot (bvlshr (bvnot #x0) #x1)))
(define-fun max () (BitVec 4)
    (bvnot min))
(define-fun l ((s (BitVec 4)) (t (BitVec 4))) Bool
    (bvuge (bvneg (inv s t)) t))
(define-fun SC ((s (BitVec 4)) (t (BitVec 4))) Bool
    true)
(constraint (=> (SC s t) (l s t)))

(check-synth)
Solution:
(define-fun inv ((s (BitVec 4)) (t (BitVec 4))) (BitVec 4) (bvneg t))

Question:
(set-logic BV)

(synth-fun inv ((s (BitVec 4)) (t (BitVec 4))) (BitVec 4)
    ((Start (BitVec 4)))
    ((Start (BitVec 4) (s t #x0 #x8 #x7 (bvneg Start) (bvnot Start) (bvadd Start Start) (bvsub Start Start) (bvand Start Start) (bvlshr Start Start) (bvor Start Start) (bvshl Start Start)))))

(declare-var s (BitVec 4))
(declare-var t (BitVec 4))
(define-fun udivtotal ((a (BitVec 4)) (b (BitVec 4))) (BitVec 4)
    (ite (= b #x0) #xF (bvudiv a b)))
(define-fun uremtotal ((a (BitVec 4)) (b (BitVec 4))) (BitVec 4)
    (ite (= b #x0) a (bvurem a b)))
(define-fun min () (BitVec 4)
    (bvnot (bvlshr (bvnot #x0) #x1)))
(define-fun max () (BitVec 4)
    (bvnot min))
(define-fun l ((s (BitVec 4)) (t (BitVec 4))) Bool
    (bvule (bvmul (inv s t) s) t))
(define-fun SC ((s (BitVec 4)) (t (BitVec 4))) Bool
    true)
(constraint (=> (SC s t) (l s t)))

(check-synth)
Solution:
(define-fun inv ((s (BitVec 4)) (t (BitVec 4))) (BitVec 4) #b0000)

Question:
(set-logic BV)

(synth-fun inv ((s (BitVec 4)) (t (BitVec 4))) (BitVec 4)
    ((Start (BitVec 4)))
    ((Start (BitVec 4) (s t #x0 #x8 #x7 (bvneg Start) (bvnot Start) (bvadd Start Start) (bvsub Start Start) (bvand Start Start) (bvlshr Start Start) (bvor Start Start) (bvshl Start Start)))))

(declare-var s (BitVec 4))
(declare-var t (BitVec 4))
(define-fun udivtotal ((a (BitVec 4)) (b (BitVec 4))) (BitVec 4)
    (ite (= b #x0) #xF (bvudiv a b)))
(define-fun uremtotal ((a (BitVec 4)) (b (BitVec 4))) (BitVec 4)
    (ite (= b #x0) a (bvurem a b)))
(define-fun min () (BitVec 4)
    (bvnot (bvlshr (bvnot #x0) #x1)))
(define-fun max () (BitVec 4)
    (bvnot min))
(define-fun l ((s (BitVec 4)) (t (BitVec 4))) Bool
    (= (bvlshr (inv s t) s) t))
(define-fun SC ((s (BitVec 4)) (t (BitVec 4))) Bool
    (= (bvlshr (bvshl t s) s) t))
(constraint (=> (SC s t) (l s t)))

(check-synth)
Solution:"""

end_mask = df["next_token"].isna()
df["sequence_id"] = end_mask.shift(fill_value=False).cumsum()

# parser-state dimension
P_state = len(df["parser_state_onehot"].iloc[0])

# prepare stack_onehot
sample_stack = df["stack"].iloc[0]
if all(isinstance(e, str) for e in sample_stack):
    # map state names to indices
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
        return int(np.argmax(logprobs))  # index of highest logprob


df["next_token_id"] = df.apply(next_id_or_argmax, axis=1)

# aggregate features by sequence
X_rows, y_rows = [], []

for seq_id, group in df.groupby("sequence_id"):
    # 1. Compute summed log-prob difference over the sequence
    sequence_diffs = []
    for _, row in group.iterrows():
        syncode = np.array(row["syncode_logprobs"])
        base = np.array(row["baseline_logprobs"])
        sequence_diffs.append(base - syncode)

        """logprobs = np.array(row["syncode_logprobs"])  # this is usually a list of floats
        # find valid (non -inf) entries
        valid_mask = ~np.isneginf(logprobs)  # True where not -inf
        valid_indices = np.where(valid_mask)[0]
        valid_values = logprobs[valid_mask]
        print("Valid token indices:", valid_indices)
        print("Their logprobs:", valid_values)
        """

        logprobs = np.array(row["syncode_logprobs"])  # this is usually a list of floats

    sequence_diff = np.stack(sequence_diffs).sum(axis=0)

    # 2. For each row, make a training example
    for row_idx, row in group.iterrows():
        # Construct features for the current row
        state_vec = np.asarray(row["parser_state_onehot"], dtype=float)
        stack_vec = stack_onehot(row["stack"])
        rem_len = len(row["remainder"])
        x = np.concatenate([state_vec, stack_vec, [rem_len]])
        X_rows.append(x)

        # Get next_token_id and use it to pick value from sequence_diff
        next_id = row["next_token_id"]

        val = sequence_diff[next_id]

        if np.isinf(val):
            print(f"[WARNING] seq_id={seq_id}, row_index={row_idx}, next_id={next_id}, value is inf")
        if np.isposinf(val):
            val = 1e6
        elif np.isneginf(val):
            val = -1e6
        y_rows.append(val)

# Final arrays
X = np.vstack(X_rows)
y = np.array(y_rows)

model = LinearRegression()
model.fit(X, y)
w = model.coef_
w_state = w[:P_state]
w_stack = w[P_state:P_state + P_stack]
w_rem = w[-1]
w_K = model.intercept_


def f(s_onehot, stack_idxs, rem):
    v_state = np.asarray(s_onehot, dtype=float).dot(w_state)
    v_stack = stack_onehot(stack_idxs).dot(w_stack)
    v_rem = w_rem * len(rem)
    return float(v_state + v_stack + v_rem + w_K)


processor = SyncodeWithFProcessor(
    grammar_text=grammar_text,
    tokenizer=tokenizer,
    parser_state_extractor_cls=ParserStateExtractor,
    f_shift_fn=f,
    syncode_proc=syncode_proc,
    stack_context_length=3
)

# with open("gadprompts.txt", 'r') as file:
# valid_sequences = [line.strip() for line in file if line.strip()]

# the below approached failed due to OOM when loading the model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto",
                                             torch_dtype=getattr(torch, "float16"))
model.eval()

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

batch_size = 10
num_batches = (number_of_samples + batch_size - 1) // batch_size
llm_counts = {}

for batch_idx in range(num_batches):
    curr_batch_size = min(batch_size, number_of_samples - batch_idx * batch_size)
    try:
        outputs = model.generate(
            input_ids=input_ids.repeat(curr_batch_size, 1),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
        )
    except Exception as e:
        print(e)
        outputs = {}

    for seq_tensor in outputs:
        text = tokenizer.decode(seq_tensor, skip_special_tokens=True)
        raw_response = text.replace(prompt, "").strip()
        raw_response = re.sub(r"\s+", " ", raw_response)

        is_valid = check_validity_of_output(parser, raw_response)
        key = raw_response if is_valid else f"[INVALID] {raw_response}"

        llm_counts[key] = llm_counts.get(key, 0) + 1

total = sum(llm_counts.values())
p_llm = {seq: count / total for seq, count in llm_counts.items()}

# separate valid and invalid distributions
p_llm_valid = {seq: p for seq, p in p_llm.items() if not seq.startswith("[INVALID]")}
p_llm_invalid = {seq: p for seq, p in p_llm.items() if seq.startswith("[INVALID]")}

print("\nNormalized LLM distribution (including invalids):")
for seq, prob in sorted(p_llm.items()):
    print(f"{seq}: {prob:.6f}")

Z_valid = sum(p_llm_valid.values())
px_given_alpha = {seq: prob / Z_valid for seq, prob in p_llm_valid.items()}

print("\n[Sampled] px_given_alpha from LLM generations:")
for seq, prob in px_given_alpha.items():
    print(f"{seq}: {prob:.6f}")

# generate samples using Syncode
syncode = Syncode(model="Qwen/Qwen3-4B",
                  grammar=grammar_text,
                  parse_output_only=True, do_sample=True,
                  pad_token_id=tokenizer.eos_token_id,
                  eos_token_id=tokenizer.eos_token_id,
                  max_new_tokens=MAX_NEW_TOKENS,
                  top_p=TOP_P,
                  top_k=TOP_K,
                  temperature=TEMPERATURE, )
syncode_counts = {}
syncode_invalid_counts = {}

for _ in range(number_of_samples):
    out = syncode.infer(prompt=prompt)
    output = out[0].strip()
    is_valid = check_validity_of_output(parser, output)
    key = output if is_valid else f"[INVALID] {output}"
    syncode_counts[key] = syncode_counts.get(key, 0) + 1

total_syncode = sum(syncode_counts.values())
p_syncode_all = {seq: count / total_syncode for seq, count in syncode_counts.items()}

# separate valid and invalid
p_syncode = {seq: prob for seq, prob in p_syncode_all.items() if not seq.startswith("[INVALID]")}
p_syncode_invalid = {seq: prob for seq, prob in p_syncode_all.items() if seq.startswith("[INVALID]")}

print("\nSyncode raw counts:")
for seq, count in syncode_counts.items():
    print(f"{seq}: {count}")

print("p_syncode:")
for seq, prob in sorted(p_syncode.items()):
    print(f"{seq}: {prob:.6f}")

# generate samples from our model?
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,  # makes attention_mask
    truncation=False,
).to(model.device)

# if for some reason attention_mask is missing (older tokenizers), create it:
if "attention_mask" not in inputs:
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
processors = LogitsProcessorList([processor])

our_counts = {}
num_batches = (number_of_samples + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    curr_batch_size = min(batch_size, number_of_samples - batch_idx * batch_size)
    try:
        outputs = model.generate(
            input_ids=inputs["input_ids"].repeat(curr_batch_size, 1),
            attention_mask=inputs["attention_mask"].repeat(curr_batch_size, 1),
            do_sample=True,
            top_p=TOP_P,
            top_k=TOP_K,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            logits_processor=processors,
            repetition_penalty=REPETITION_PENALTY,
        )
    except Exception as e:
        print("Generation error:", e)
        outputs = []

    for seq_tensor in outputs:
        text = tokenizer.decode(seq_tensor, skip_special_tokens=True)
        raw_response = text.replace(prompt, "").strip()
        raw_response = re.sub(r"\s+", " ", raw_response)
        is_valid = check_validity_of_output(parser, raw_response)

        key = raw_response if is_valid else f"[INVALID] {raw_response}"

        our_counts[key] = our_counts.get(key, 0) + 1

total = sum(our_counts.values())
p_ours_all = {seq: count / total for seq, count in our_counts.items()}

# Separate valid/invalid
p_ours = {seq: prob for seq, prob in p_ours_all.items() if not seq.startswith("[INVALID]")}
p_ours_invalid = {seq: prob for seq, prob in p_ours_all.items() if seq.startswith("[INVALID]")}

print("\nOurs raw counts:")
for seq, count in our_counts.items():
    print(f"{seq}: {count}")

print("\nNormalized Our distribution):")
for seq, prob in sorted(p_ours.items()):
    print(f"{seq}: {prob:.6f}")


# KL divergence to ground truth
def kl_div(p, q, epsilon=1e-12):
    """
    Computes KL(p || q) where p and q are dicts representing probability distributions.
    Adds epsilon smoothing to avoid division by zero or log(0).
    Prints any keys that are only in one of the distributions.
    """

    all_keys = set(p.keys()).union(q.keys())
    only_in_p = set(p.keys()) - set(q.keys())
    only_in_q = set(q.keys()) - set(p.keys())

    if only_in_p:
        print("[KL WARNING] Keys in p but not in q:", sorted(only_in_p))
    if only_in_q:
        print("[KL WARNING] Keys in q but not in p:", sorted(only_in_q))

    total = 0.0
    for x in all_keys:
        px = p.get(x, 0.0)
        qx = q.get(x, 0.0)

        # Add epsilon smoothing
        px = max(px, epsilon)
        qx = max(qx, epsilon)

        total += px * np.log(px / qx)

    return total


kl_syncode = kl_div(px_given_alpha, p_syncode)
kl_ours = kl_div(px_given_alpha, p_ours)
kl_syncode_ours = kl_div(p_syncode, p_ours)

print(f"KL(GT||Syncode): {kl_syncode:.4f}")
print(f"KL(GT||Ours): {kl_ours:.4f}")
print(f"KL(Syncode||Ours): {kl_syncode_ours:.4f}")

all_sequences = set(llm_counts.keys()) | set(syncode_counts.keys()) | set(our_counts.keys())
valid_sequences = {seq for seq in all_sequences if not seq.startswith("[INVALID]")}
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


plot_distribution(px_given_alpha, "LLM", "llm_distribution.png")
plot_distribution(p_syncode, "SynCode", "syncode_distribution.png")
plot_distribution(p_ours, "Ours", "ours_distribution.png")

invalid_keys = sorted(
    set(p_llm_invalid.keys()) | set(p_syncode_invalid.keys()) | set(p_ours_invalid.keys())
)
x_invalid = np.arange(len(invalid_keys))


def plot_invalid_distribution(probs, method_name, filename):
    plt.figure(figsize=(15, 7), constrained_layout=True)
    plt.bar(x_invalid, [probs.get(seq, 0.0) for seq in invalid_keys], width=0.6)
    plt.xticks(x_invalid, invalid_keys, rotation=90, ha='left')  # better anchoring
    plt.xlabel("Invalid Sequences")
    plt.ylabel("Probability")
    plt.title(f"{method_name} Distribution over Invalid Sequences")

    plt.savefig(f"plots/{filename}")
    plt.close()


plot_invalid_distribution(p_llm_invalid, "LLM", "llm_invalid_distribution.png")
plot_invalid_distribution(p_syncode_invalid, "SynCode", "syncode_invalid_distribution.png")
plot_invalid_distribution(p_ours_invalid, "Ours", "ours_invalid_distribution.png")

output_data = {
    "valid": {
        "px_given_alpha": px_given_alpha,
        "p_syncode": p_syncode,
        "p_ours": p_ours
    },
    "invalid": {
        "p_llm_invalid": p_llm_invalid,
        "p_syncode_invalid": p_syncode_invalid,
        "p_ours_invalid": p_ours_invalid
    },
    "kl_divergences": {
        "KL(GT||Syncode)": f"{kl_syncode:.4f}",
        "KL(GT||Ours)": f"{kl_ours:.4f}",
        "KL(Syncode||Ours)": f"{kl_syncode_ours:.4f}"
    }
}

os.makedirs("results", exist_ok=True)
output_path = "results/distribution_data.json"
with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)
