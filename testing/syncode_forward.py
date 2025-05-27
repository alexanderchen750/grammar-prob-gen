from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from syncode import SyncodeLogitsProcessor, Grammar
import os

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto", torch_dtype=torch.float16)
model.eval()

grammar_file = "json.lark"

with open(grammar_file, 'r') as f:
    grammar = f.read()


json_grammar = Grammar(grammar)
syncode_logits_processor = SyncodeLogitsProcessor(grammar=json_grammar, tokenizer=tokenizer, parse_output_only=True)

# Define prompt and known output
prompt = "Generate a person JSON.\n"
output = '  { "name": "John Doe", "age": 30, "gender": "male" }'
full_text = output

# Tokenize
input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

# Run forward pass
with torch.inference_mode():
    logits = model(input_ids).logits  # [1, seq_len, vocab_size]

# Apply processor step-by-step
print(" Full vocabulary predictions at each step (with Syncode constraints):\n")
syncode_logits_processor.reset()

with open("logits_output_syncode.txt", "w") as f:
    for i in range(len(input_ids[0]) - 1):
        prev_ids = input_ids[:, :i+1]  # up to and including token i

        # Raw logits at this step
        logits_i = logits[0, i]  # shape: [vocab_size]

        # Wrap to shape [1, vocab_size] and process
        logits_processed = syncode_logits_processor(prev_ids, logits_i.unsqueeze(0).clone())[0]

        probs = F.softmax(logits_processed, dim=-1)

        token_id_prev = input_ids[0, i].item()
        token_id_target = input_ids[0, i + 1].item()

        topk = torch.topk(probs, k=10)
        f.write(f"Step {i+1} (after token: {tokenizer.decode([token_id_prev])!r})\n")
        f.write(f"  → Actual next token: {tokenizer.decode([token_id_target])!r} (P = {probs[token_id_target].item():.6f})\n")

        for j in range(10):
            tid = topk.indices[j].item()
            prob = topk.values[j].item()
            token_str = tokenizer.decode([tid])
            mark = " ←" if tid == token_id_target else ""
            f.write(f"     {token_str!r:<15} P = {prob:.6f}{mark}\n")
        f.write("-" * 50 + "\n")
