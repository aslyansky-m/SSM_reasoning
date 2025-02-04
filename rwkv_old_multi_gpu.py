import os
os.environ['HF_HOME'] = '/worxpace/dev/nlp/cache/'
os.environ['HF_HUB_CACHE'] = '/worxpace/dev/nlp/cache/'
os.environ['TRANSFORMERS_CACHE'] = '/worxpace/dev/nlp/cache/'

import torch
from dataset import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_name = "RWKV/rwkv-raven-7b"

model_rwkv = AutoModelForCausalLM.from_pretrained(model_name).to('cpu')
model_rwkv_gpu = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(0)
tokenizer_rwkv = AutoTokenizer.from_pretrained(model_name)

if tokenizer_rwkv.pad_token is None:
    tokenizer_rwkv.pad_token = tokenizer_rwkv.eos_token

def get_rwkv_output(input_text: str, max_new_tokens: int = 40):
    input_ids = tokenizer_rwkv(input_text, return_tensors="pt")["input_ids"]
    try:
        out = model_rwkv_gpu.generate(input_ids.to(0), max_new_tokens=max_new_tokens)
        return tokenizer_rwkv.batch_decode(out)[0]
    except:
        out = model_rwkv.generate(input_ids.to('cpu'), max_new_tokens=max_new_tokens)
        return tokenizer_rwkv.batch_decode(out)[0]

get_rwkv_output("Hey how are you doing?")



max_new_tokens = 40
prompts = get_prompts()
results = []

for i, prompt in enumerate(tqdm(prompts)):
    input_ids = tokenizer_rwkv(prompt, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] < 1024:
        out = model_rwkv_gpu.generate(input_ids.to(0), max_new_tokens=max_new_tokens)
        out = tokenizer_rwkv.batch_decode(out)[0]
    else:
        continue
        out = model_rwkv.generate(input_ids.to('cpu'), max_new_tokens=max_new_tokens)
        out = tokenizer_rwkv.batch_decode(out)[0]
    results.append([i,out])

import pickle
with open("results/rwkv.pkl", "wb") as f:
    pickle.dump(results, f)
    
