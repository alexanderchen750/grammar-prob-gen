import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
from syncode import Syncode
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_pickle("training_data/grammar_data_df.pkl")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
prompt = "Generate a random sequence of 5 binary digits (0 or 1):"

df["sequence_id"] = (df.index // 5).astype(int)

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
    for _, row in group.iterrows():
        # Construct features for the current row
        state_vec = np.asarray(row["parser_state_onehot"], dtype=float)
        stack_vec = stack_onehot(row["stack"])
        rem_len = len(row["remainder"])
        x = np.concatenate([state_vec, stack_vec, [rem_len]])
        X_rows.append(x)

        # Get next_token_id and use it to pick value from sequence_diff
        next_id = row["next_token_id"]
        y_rows.append(sequence_diff[next_id])

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


with open("gadprompts.txt", 'r') as file:
    valid_sequences = [line.strip() for line in file if line.strip()]

# the below approached failed due to OOM when loading the model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto",
                                             torch_dtype=getattr(torch, "float16"))
model.eval()

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# generate sequences from the LLM
num_samples = 1000
outputs = model.generate(
    input_ids=input_ids.repeat(num_samples, 1),
    max_new_tokens=5,
    do_sample=True,
    top_p=0.95,
    temperature=1.0,
    pad_token_id=tokenizer.eos_token_id
)

llm_counts = {}
for seq_tensor in outputs:
    text = tokenizer.decode(seq_tensor, skip_special_tokens=True)
    sequence = text.replace(prompt, "").strip()
    key = sequence if sequence in valid_sequences else f"[INVALID] {sequence}"
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
for seq in sorted(valid_sequences):
    print(f"{seq}: {px_given_alpha.get(seq, 0.0):.6f}")

# generate samples using Syncode
with open("grammars/gad.lark", 'r') as file:
    grammar_text = file.read()

syncode = Syncode(model="Qwen/Qwen3-4B", grammar=grammar_text, parse_output_only=True)
syncode_counts = {seq: 0 for seq in valid_sequences}
syncode_invalid_counts = {}
num_samples = 1000

for _ in range(num_samples):
    out = syncode.infer(prompt=prompt)
    output = out[0].strip()
    key = output if output in valid_sequences else f"[INVALID] {output}"
    syncode_counts[key] = syncode_counts.get(key, 0) + 1

total_syncode = sum(syncode_counts.values())
p_syncode_all = {seq: count / total_syncode for seq, count in syncode_counts.items()}

# separate valid and invalid
p_syncode = {seq: prob for seq, prob in p_syncode_all.items() if not seq.startswith("[INVALID]")}
p_syncode_invalid = {seq: prob for seq, prob in p_syncode_all.items() if seq.startswith("[INVALID]")}

print("\nSyncode raw counts:")
for seq in valid_sequences:
    print(f"{seq}: {syncode_counts[seq]}")


print("p_syncode:")
for seq, prob in sorted(p_syncode.items()):
    print(f"{seq}: {prob:.6f}")


# generate samples from our model?
def sample_from_ours(df, f, tokenizer, num_samples=1000):
    counts = {}

    for _ in range(num_samples):
        row0 = df[df["prefix_text"] == "0"].iloc[0]
        row1 = df[df["prefix_text"] == "1"].iloc[0]

        probs = []
        for row in [row0, row1]:
            shift = f(row["parser_state_onehot"], row["stack"], row["remainder"])
            logits = np.array(row["syncode_logprobs"])
            adjusted = logits + shift
            probs.append(adjusted)

        probs = np.vstack(probs)  # shape (2, V)
        probs = np.exp(probs - np.max(probs, axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)

        p0 = probs[0][tokenizer.encode("0", add_special_tokens=False)[0]]
        p1 = probs[1][tokenizer.encode("1", add_special_tokens=False)[0]]
        p = np.array([p0, p1])
        p = p / p.sum()

        generated = np.random.choice(["0", "1"], p=p)

        for step in range(4):
            matching_rows = df[df["prefix_text"] == generated]
            if matching_rows.empty:
                break

            row = matching_rows.iloc[0]
            logits = np.array(row["syncode_logprobs"])
            shift = f(row["parser_state_onehot"], row["stack"], row["remainder"])
            adjusted_logits = logits + shift
            probs = np.exp(adjusted_logits - np.max(adjusted_logits))
            probs = probs / probs.sum()

            next_token_id = np.random.choice(len(probs), p=probs)
            next_token = tokenizer.decode([next_token_id]).strip()
            if next_token not in ["0", "1"]:
                break

            generated += next_token

        key = generated if generated in valid_sequences else f"[INVALID] {generated}"
        counts[key] = counts.get(key, 0) + 1

    return counts



our_counts = sample_from_ours(df, f, tokenizer, num_samples=1000)
total = sum(our_counts.values())
p_ours_all = {seq: count / total for seq, count in our_counts.items()}

# Separate valid/invalid
p_ours = {seq: prob for seq, prob in p_ours_all.items() if not seq.startswith("[INVALID]")}
p_ours_invalid = {seq: prob for seq, prob in p_ours_all.items() if seq.startswith("[INVALID]")}



print("\nOurs raw counts:")
for seq in valid_sequences:
    print(f"{seq}: {our_counts[seq]}")

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

print(f"KL(Syncode || ground-truth): {kl_syncode:.4f}")
print(f"KL(Ours   || ground-truth): {kl_ours:.4f}")
print(f"KL(Ours   || syncode): {kl_syncode_ours:.4f}")

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
    plt.figure(figsize=(15, 5))
    plt.bar(x_invalid, [probs.get(seq, 0.0) for seq in invalid_keys], width=0.6)
    plt.xticks(x_invalid, invalid_keys, rotation=90)
    plt.xlabel("Invalid Sequences")
    plt.ylabel("Probability")
    plt.title(f"{method_name} Distribution over Invalid Sequences")
    plt.tight_layout()
    plt.savefig(f"plots/{filename}")
    plt.close()


plot_invalid_distribution(p_llm_invalid, "LLM", "llm_invalid_distribution.png")
plot_invalid_distribution(p_syncode_invalid, "SynCode", "syncode_invalid_distribution.png")
plot_invalid_distribution(p_ours_invalid, "Ours", "ours_invalid_distribution.png")
