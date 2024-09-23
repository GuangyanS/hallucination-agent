import pickle
import re
import numpy as np
import json
from tqdm import tqdm
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('/home/gs3260/init_weights/Meta-Llama-3.1-8B-Instruct', device_map="auto")
model = AutoModelForCausalLM.from_pretrained('/home/gs3260/init_weights/Meta-Llama-3.1-8B-Instruct', device_map="auto", torch_dtype=torch.float16)


def parse_reflection(reflection_text):
    """
    Parses the reflection text to extract 'Believable' and 'Revised Answer'.
    """
    terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    messages = [
                {"role": "system", "content": "Extract the value assigned to revised_answer if the believable is false. Don't answer anything more than a word. If believable factor is true or there is no revised answer, please write 'None'."},
                {"role": "user", "content": reflection_text},
            ]
    input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

    outputs = model.generate(input_ids, 
                            pad_token_id=tokenizer.eos_token_id,
                            return_dict_in_generate=True, 
                            output_scores=True, 
                            max_new_tokens=20,
                            eos_token_id=terminators,
                            do_sample=False,
                            )
    response = tokenizer.batch_decode(outputs.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    response = response.strip()
    if response.lower() == "none":
        return True, None
    else:
        return False, response

def compute_confidence(token_log_likelihoods):
    """
    Calculates the average log probability as a measure of confidence.
    """
    if not token_log_likelihoods:
        return None
    avg_log_prob = sum(token_log_likelihoods) / len(token_log_likelihoods)
    confidence = math.exp(avg_log_prob)
    return confidence

# Load the data from the pickle file
with open('experiment/commonsensqa.pkl', 'rb') as file:
    data = pickle.load(file)

# Check if data is a list or a dictionary
if isinstance(data, dict):
    data_items = data.items()
elif isinstance(data, list):
    data_items = enumerate(data)
else:
    raise TypeError('Data should be a list or a dictionary.')

# Process each data point
for key, values in tqdm(data_items, desc="Processing data points"):
    # Extract 'believable' and 'revised_pred' from 'reflection'
    reflection_text = values.get('reflection', '')
    believable, revised_pred = parse_reflection(reflection_text)
    values['believable'] = believable
    values['revised_pred'] = revised_pred
    print(f"Processed data point {key} with revised prediction: {revised_pred}")

    # Compute 'confidence' from 'token_log_likelihoods'
    token_log_likelihoods = values.get('token_log_likelihoods', [])
    confidence = compute_confidence(token_log_likelihoods)
    values['confidence'] = confidence

# Save the updated data to a new pickle file
with open('experiment/modi_commonsensqa.pkl', 'wb') as f:
    pickle.dump(data, f)