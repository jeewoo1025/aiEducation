from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# huggingface: https://huggingface.co/meta-llama/Meta-Llama-3-8B
model_name = 'meta-llama/Meta-Llama-3-8B'

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir='.')
token = AutoTokenizer.from_pretrained(model_name)

# save path
model.save_pretrained('Meta-Llama-3-8B')
token.save_pretrained('Meta-Llama-3-8B')
