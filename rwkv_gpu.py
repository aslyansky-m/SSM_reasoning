import os
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

model = RWKV(model='/worxpace/dev/nlp/cache/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth', strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) 


from dataset import *
prompts = get_prompts()


from tqdm import tqdm
results = []
for p in tqdm(prompts[:100]):
    res = pipeline.generate(p, token_count=200, args=args)
    results.append(res)