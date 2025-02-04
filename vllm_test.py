from vllm import LLM, SamplingParams

import os
os.environ["HF_TOKEN"] = None

from dataset import get_prompts

# Sample prompts.
prompts = get_prompts()
prompts = prompts[:4000]
# Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
sampling_params = SamplingParams(temperature=0, top_k=1)

# Create an LLM.
models_dict = {
    "jamba": "ai21labs/AI21-Jamba-1.5-Mini",
    "lamma-3.1i": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2"
}

for model_name, model in models_dict.items():
    llm = LLM(model=model,tensor_parallel_size=4)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    import pickle
    with open(f'results/{model_name}_greedy.pkl', 'wb') as f:
        pickle.dump(outputs, f)

