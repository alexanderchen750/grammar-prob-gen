import torch
from syncode import SyncodeLogitsProcessor
from syncode import Grammar
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

device = 'cuda'
model_name = "Qwen/Qwen3-4B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


grammar_file = "json.lark"
with open(grammar_file, 'r') as f:
    grammar_json = f.read()
grammar=Grammar(grammar_json)
print("creating syncode logits processor")
syncode_logits_processor = SyncodeLogitsProcessor(grammar=grammar, tokenizer=tokenizer, parse_output_only=True)

prompt = "Generate a json for the profile of a person, with values for name, age, and gender\n"
syncode_logits_processor.reset()

inputs = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

print("generate")
output = model.generate(
    inputs,
    max_length=100, 
    num_return_sequences=1, 
    pad_token_id=tokenizer.eos_token_id, 
    logits_processor=[syncode_logits_processor]
    )
output_str = tokenizer.decode(output[0][len(inputs[0]):], skip_special_tokens=True)
print(output_str)