import os
os.environ['HF_HOME'] = '/worxpace/dev/nlp/cache/'
os.environ['HF_HUB_CACHE'] = '/worxpace/dev/nlp/cache/'
os.environ['TRANSFORMERS_CACHE'] = '/worxpace/dev/nlp/cache/'

import torch
from dataset import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from tqdm import tqdm



model_name = "RWKV/rwkv-raven-7b"

# Load model and tokenizer outside the parallel loop
model_rwkv = AutoModelForCausalLM.from_pretrained(model_name).to('cpu')
tokenizer_rwkv = AutoTokenizer.from_pretrained(model_name)

if tokenizer_rwkv.pad_token is None:
    tokenizer_rwkv.pad_token = tokenizer_rwkv.eos_token

max_new_tokens = 40
prompts = get_prompts()

def process_prompt(prompt):
    input_ids = tokenizer_rwkv(prompt, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] <= 1024:
        return ' '
    out = model_rwkv.generate(input_ids.to('cpu'), max_new_tokens=max_new_tokens)
    out = tokenizer_rwkv.batch_decode(out)[0]
    return out

results = []

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_prompt, prompt): i for i, prompt in enumerate(prompts)}
    for future in tqdm(as_completed(futures), total=len(prompts)):
        i = futures[future]
        try:
            result = future.result()
            results.append([i, result])
        except Exception as e:
            print(f"Prompt {i} generated an exception: {e}")

# Save results
with open("results/rwkv_cpu.pkl", "wb") as f:
    pickle.dump(results, f)
