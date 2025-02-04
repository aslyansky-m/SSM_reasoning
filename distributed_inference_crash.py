import os
import pickle
import random
import time
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object, tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import get_prompts

import os
os.environ['HF_HOME'] = '/worxpace/dev/nlp/cache/'
os.environ['HF_HUB_CACHE'] = '/worxpace/dev/nlp/cache/'
os.environ['TRANSFORMERS_CACHE'] = '/worxpace/dev/nlp/cache/'

accelerator = Accelerator()
prompts = get_prompts()

token = None

# Model and tokenizer paths
model_path = "tiiuae/falcon-mamba-7b"
output_dir = "/worxpace/dev/nlp/results/falcon-mamba-7b/"
os.makedirs(output_dir, exist_ok=True)

# Each process gets a unique sub-folder for its results
process_dir = os.path.join(output_dir, f"process_{accelerator.process_index}")
os.makedirs(process_dir, exist_ok=True)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.float32,
    token=token
)
tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)   
tokenizer.pad_token = tokenizer.eos_token

# Prepare prompts in batches
def prepare_prompts(prompts, tokenizer, batch_size=16):
    # Create batches of prompts with global indices
    batches = [
        prompts[i:i + batch_size]  # Each batch is a tuple (global_index, batch_prompts)
        for i in range(0, len(prompts), batch_size)
    ]

    # Shuffle the batches
    random.shuffle(batches)

    # Tokenize batches
    batches_tok = []
    tokenizer.padding_side = "left"  # Adjust padding side for inference
    for batch in batches:
        # Ensure batch_prompts is a List[str]
        batch_ids = [prompt[0] for prompt in batch]  # Extract prompt ID
        batch_texts = [prompt[1] for prompt in batch]  # Extract prompt text
        tokenized_batch = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding='longest', 
            truncation=False, 
            pad_to_multiple_of=8,
            add_special_tokens=False
        ).to("cuda") 
        batches_tok.append([batch_ids[0], tokenized_batch])
    tokenizer.padding_side = "right"  # Restore default padding side
    return batches_tok

# Save results for a single batch with its global index
def save_batch_results(process_dir, global_index, prompts, outputs):
    batch_file = os.path.join(process_dir, f"batch_{global_index}.pkl")
    with open(batch_file, "wb") as f:
        pickle.dump({"global_index": global_index, "prompts": prompts, "outputs": outputs}, f)

# Sync GPUs and start timer
accelerator.wait_for_everyone()    
start = time.time()

# Attach indices to prompts
prompts_all = [(i, x) for i, x in enumerate(prompts)]

# Split prompts between processes
with accelerator.split_between_processes(prompts_all) as prompts:
    prompt_batches = prepare_prompts(prompts, tokenizer, batch_size=4)

    for global_index, prompts_tokenized in tqdm(prompt_batches):
        try:
            batch_file = os.path.join(process_dir, f"batch_{global_index}.pkl")
            if os.path.exists(batch_file):
                continue
            
            outputs_tokenized = model.generate(
                **prompts_tokenized, 
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id
            )

            # Remove prompt from generated tokens
            outputs_tokenized = [
                tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)
            ]           
            

            # Decode the generated outputs
            outputs = tokenizer.batch_decode(outputs_tokenized)

            # Save prompts and results immediately with global index
            save_batch_results(process_dir, global_index, prompts_tokenized, outputs)

        except Exception as e:
            print(f"Error processing batch {global_index}: {e}")
            continue

# Gather results if necessary
results_gathered = gather_object([])  # Placeholder for cross-process collection

# Final processing (main process)
if accelerator.is_main_process:
    timediff = time.time() - start
    print(f"Processing complete. Time elapsed: {timediff}s")
