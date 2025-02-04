import os
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1'



from accelerate import Accelerator
from accelerate.utils import gather_object, tqdm
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from statistics import mean
import torch, time


output_path = "results/rwkv_v6.pkl"
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
prompts_all = get_prompts()
prompts_all = [(i,x) for i,x in enumerate(prompts_all)]


accelerator = Accelerator()


# sync GPUs and start the timer
accelerator.wait_for_everyone()    
start=time.time()


# divide the prompt list onto the available GPUs 
with accelerator.split_between_processes(prompts_all) as prompts:
    results=dict(outputs=[], num_tokens=0)


    for ind, proompt in tqdm(prompts):
        
        outputs = pipeline.generate(proompt, token_count=200, args=args)

        # store in results{} to be gathered by accelerate
        results["outputs"].extend([ind,outputs])
        results["num_tokens"] += 200

    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

results_gathered=gather_object(results)


if accelerator.is_main_process:
    import pickle
    with open(output_path, "wb") as f:
        pickle.dump(results_gathered, f)
    timediff=time.time()-start
    num_tokens=sum([r["num_tokens"] for r in results_gathered ])

    print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")