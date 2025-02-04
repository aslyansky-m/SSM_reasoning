import os
os.environ['HF_HOME'] = '/worxpace/dev/nlp/cache/'
os.environ['HF_HUB_CACHE'] = '/worxpace/dev/nlp/cache/'
os.environ['TRANSFORMERS_CACHE'] = '/worxpace/dev/nlp/cache/'

from accelerate import Accelerator
from accelerate.utils import gather_object, tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json

from dataset import *
prompts = get_prompts()


token = None

accelerator = Accelerator()

def write_pretty_json(file_path, data):
    import json
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)

prompts_all = prompts

batch_size = 4
# load a base model and tokenizer
# model_path="meta-llama/Llama-2-7b-hf"
# model_path = "CobraMamba/mamba-gpt-7b"
# output_path = "results/mamba.pkl"
model_path ="tiiuae/falcon-mamba-7b"
output_path = "results/falcon-mamba-7b.pkl"
# model_path = "state-spaces/mamba2-2.7b"
# output_path = "results/mamba2-2.7b.pkl"


model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
    token=token
)
tokenizer = AutoTokenizer.from_pretrained(model_path,token=token)   
tokenizer.pad_token = tokenizer.eos_token

# batch, left pad (for inference), and tokenize
def prepare_prompts(prompts, tokenizer, batch_size=16):
    batches = [[
        [prompt[0] for prompt in prompts[i:i + batch_size]], 
        [prompt[1] for prompt in prompts[i:i + batch_size]]]
        for i in range(0, len(prompts), batch_size)
    ]

    batches_tok=[]
    tokenizer.padding_side="left"     
    for ind, prompt_batch in batches:
        batches_tok.append([
            ind,
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            ])
    tokenizer.padding_side="right"
    return batches_tok

# sync GPUs and start the timer
accelerator.wait_for_everyone()    
start=time.time()

prompts_all = [(i,x) for i,x in enumerate(prompts_all)]

# pbar=tqdm(total=len(prompts_all))   

# divide the prompt list onto the available GPUs 
with accelerator.split_between_processes(prompts_all) as prompts:
    results=dict(outputs=[], num_tokens=0)

    # have each GPU do inference in batches
    prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=batch_size)

    for ind, prompts_tokenized in tqdm(prompt_batches):
        outputs_tokenized=model.generate(
            **prompts_tokenized, 
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id)

        # remove prompt from gen. tokens
        outputs_tokenized=[ tok_out[len(tok_in):] 
            for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

        # count and decode gen. tokens 
        num_tokens=sum([ len(t) for t in outputs_tokenized ])
        outputs=tokenizer.batch_decode(outputs_tokenized)

        # store in results{} to be gathered by accelerate
        results["outputs"].extend(zip(ind,outputs))
        results["num_tokens"] += num_tokens

    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

results_gathered=gather_object(results)


if accelerator.is_main_process:
    import pickle
    with open(output_path, "wb") as f:
        pickle.dump(results_gathered, f)
    timediff=time.time()-start
    num_tokens=sum([r["num_tokens"] for r in results_gathered ])

    print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")