import os
import random
import google.generativeai as genai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"random seed = {seed}")

def generate_response(prompt, model_name, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def tokenize(tokenizer, prompt, add_eos_token=True):
    """
    Tokenizes the input prompt and adds an end-of-sequence token if specified.
    """
    tokens = tokenizer(prompt,
                       max_length=tokenizer.model_max_length, # use the default max length
                       truncation=True,
                       padding=False,
                       
                       )
    if add_eos_token:
        tokens += [tokenizer.eos_token_id]
    return tokens