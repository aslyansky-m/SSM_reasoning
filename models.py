import os
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
os.environ['HF_HOME'] = '/worxpace/dev/nlp/cache/'
os.environ['HF_HUB_CACHE'] = '/worxpace/dev/nlp/cache/'
os.environ['TRANSFORMERS_CACHE'] = '/worxpace/dev/nlp/cache/'

token = None

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device ='cuda:0'


model_path = 'meta-llama/Llama-2-7b-hf'
tokenizer_llama = AutoTokenizer.from_pretrained(model_path, token=token)
model_llama = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto', token=token).to(0)

def get_llama_output(input_text: str, max_new_tokens: int = 32):
    input_ids = tokenizer_llama(input_text, return_tensors="pt")["input_ids"].to(0)
    out = model_llama.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer_llama.batch_decode(out)[0]


model_name = "CobraMamba/mamba-gpt-7b"

tokenizer_mamba = AutoTokenizer.from_pretrained(model_name)
model_mamba = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(0)


def get_mamba_output(input_text: str, max_length: int = 1024*5):
    input_ids = tokenizer_mamba.encode(input_text, return_tensors="pt").to(0)
    output = model_mamba.generate(input_ids, max_length=max_length, temperature=0.7)
    output_text = tokenizer_mamba.decode(output[0], skip_special_tokens=True)
    return output_text

rwkv_model = RWKV(model='/worxpace/dev/nlp/cache/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth', strategy='cuda:0 fp16')
rwkv_pipeline = PIPELINE(rwkv_model, "rwkv_vocab_v20230424")

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) 

def get_rwkv_output(input_text: str):
    out = rwkv_pipeline.generate(input_text, token_count=200, args=args)
    return out

model_name = "RWKV/rwkv-raven-7b"

model_rwkv = AutoModelForCausalLM.from_pretrained(model_name).to('cpu')
model_rwkv_gpu = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(0)
tokenizer_rwkv = AutoTokenizer.from_pretrained(model_name)


def get_rwkv_old_output(input_text: str, max_new_tokens: int = 40):
    input_ids = tokenizer_rwkv(input_text, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] < 1024:
        out = model_rwkv_gpu.generate(input_ids.to(0), max_new_tokens=max_new_tokens)
        return tokenizer_rwkv.batch_decode(out)[0]
    else:
        out = model_rwkv.generate(input_ids.to('cpu'), max_new_tokens=max_new_tokens)
        return tokenizer_rwkv.batch_decode(out)[0]



