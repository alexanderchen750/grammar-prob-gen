from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto", torch_dtype=torch.float16)
model.eval()

# Define prompt and known output
prompt = "Generate a person JSON.\n"
output = '  { "name": "John Doe", "age": 30, "gender": "male" }'
full_text = output

# Tokenize
input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

# Run forward pass
with torch.inference_mode():
    logits = model(input_ids).logits  # shape: [1, seq_len, vocab_size]

# Analyze token-by-token predictions
print(" Full vocabulary predictions at each step (top 10 shown):\n")
with open("logits_output_base1.txt", "w") as f:
    for i in range(len(input_ids[0]) - 1):
        token_id_prev = input_ids[0, i].item()
        token_id_target = input_ids[0, i + 1].item()

        logits_i = logits[0, i]  # logits after token i
        probs = F.softmax(logits_i, dim=-1)

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
