import string, re, os
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

dataset_url = 'https://github.com/alonj/Same-Task-More-Tokens/raw/main/FLenQA.zip'
data = pd.read_json(dataset_url, orient='records', lines=True)

prompt_structures = {

"Simplified RuleTaker": 
    lambda sample: f"""\
Answer whether the statement {sample['assertion/question']} can be derived from the rule and the facts. Answer with either "True" or "False".
Rule: {sample['rule']}
Facts: {sample['mixin']}
Answer with either "True or "False".
""",

"Simplified RuleTaker_cot": 
    lambda sample: f"""\
Answer whether the statement {sample['assertion/question']} can be derived from the rule and the facts.
Show your steps then answer with either "True" or "False".
Rule: {sample['rule']}
Facts: {sample['mixin']}
Answer with either "True or "False". Let's work this out in a step by step way to be sure we have the right answer.
""",

"PIR":
    lambda sample: f"""\
{sample['mixin']}
True/False Question: {sample['assertion/question']}
Answer only True or False.
""",

"PIR_cot":
    lambda sample: f"""\
Show your steps then answer with ’true’ or ’false’.
{sample['mixin']}
True/False Question: {sample['assertion/question']}
Let’s work this out in a step by step way to
be sure we have the right answer.
""",

"MonoRel": 
    lambda sample: f"""\
Here are some facts. Answer the exact following question based on the text: {sample['assertion/question']} Answer the question as it appears exactly.
{sample['mixin']}
{sample['assertion/question']}
Answer only True or False.
""",

"MonoRel_cot": 
    lambda sample: f"""\
Here are some facts. Answer the exact following question based on the text: {sample['assertion/question']} Answer the question as it appears exactly.
Show your steps then answer with ’true’ or ’false’.
{sample['mixin']}
{sample['assertion/question']}
Let’s work this out in a step by step way to be sure we have the right answer. Show your work and finally answer with ’true’ or ’false’. The final step should include the exact text of the question and the answer.
""",

}


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra white spaces

    Args:
    s: string to normalize

    Returns:
    normalized string
    """
    s = str(s).lower()
    s = s.replace("".join(list(set(string.punctuation))), '')
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s

def response_category(ans):
    """
    Categorize the answer as true, false or other/refused
    
    Args:
    ans: string to categorize
    
    Returns:
    string category
    """
    if isinstance(ans, (bool, np.bool_)):
        return normalize_answer(str(ans))
    if isinstance(ans, str):
        ans = normalize_answer(ans)
        ans = ans.replace('not true', 'false')
        last_true_pos = ans.rfind('true')
        last_false_pos = ans.rfind('false')
        if last_true_pos > last_false_pos:
            return 'true'
        elif last_false_pos > last_true_pos:
             return 'false'
    return 'other/refused'
    
def response_analysis(sample, response, chain_of_thought=False):
    """
    Analyze the response and compare it to the sample

    Args:
    sample: dictionary with sample information
    response: string response

    Returns:
    dictionary with analysis results
    """
    normalized_response_text = normalize_answer(response)
    categorical_response = response_category(normalized_response_text)
    correctness = categorical_response is not None and categorical_response in sample['label'].lower()
    if chain_of_thought:
        if sample['dataset'] != 'Simplified RuleTaker': # Ruletaker has statements instead of facts
            cot_coverage = sum([normalize_answer(fact) in normalized_response_text for fact in sample['facts']])
        else:
            cot_coverage = sum([normalize_answer(fact) in normalized_response_text for fact in sample['statement']])
        early_response = categorical_response is not None and categorical_response in normalized_response_text[:10].lower()
    else:
        cot_coverage = 0
        early_response = False
    return {
        'response': response,
        'cot_coverage': cot_coverage,
        'normalized_response': categorical_response,
        'correct': correctness,
        'early_response': early_response,
    }

def get_prompts():
    prompts = []
    for sample in data.to_dict(orient='records'):
        for chain_of_thought in [True, False]:
            prompt = prompt_structures[sample['dataset'] + ('_cot' if chain_of_thought else '')](sample)
            prompts.append(prompt)
    return prompts