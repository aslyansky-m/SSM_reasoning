import os
import multiprocessing as mp
import pickle

from dataset import get_prompts
from tqdm import tqdm

# Function to process a chunk of prompts on a specific GPU
def process_chunk(gpu_id, chunk, results_list):
    # Environment setup for RWKV
    os.environ['RWKV_JIT_ON'] = '1'
    os.environ["RWKV_CUDA_ON"] = '1'

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model_path = '/worxpace/dev/nlp/cache/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'
    model = RWKV(model=model_path, strategy=f'cuda:{gpu_id} fp16')
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

    args = PIPELINE_ARGS(
        temperature = 1.0, 
        top_p = 0.7, 
        top_k = 100, # top_k = 0 then ignore
        alpha_frequency = 0.25,
        alpha_presence = 0.25,
        alpha_decay = 0.996, # gradually decay the penalty
        token_ban = [0], # ban the generation of some tokens
        token_stop = [], # stop generation whenever you see any token here
        chunk_len = 256
    )

    for i, p in tqdm(chunk, desc=f"GPU {gpu_id}"):
        res = pipeline.generate(p, token_count=200, args=args)
        results_list.append((i,res))

def main():
    # Get prompts and split into 4 chunks
    prompts = get_prompts()
    prompts = prompts[12000:]
    prompts = [(i, x) for i, x in enumerate(prompts)]
    chunks = [prompts[i::4] for i in range(4)]

    # Shared list to store results
    manager = mp.Manager()
    results_list = manager.list()

    # Create a process for each GPU
    processes = []
    for gpu_id in range(4):
        p = mp.Process(target=process_chunk, args=(gpu_id, chunks[gpu_id], results_list))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Convert the shared list to a regular list and save to pickle
    results = list(results_list)
    with open('results/rwkv_v6_pt2.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
