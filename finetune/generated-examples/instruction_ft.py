# Fine-tuning an Instruction LLM (DeepSeek) on a custom dataset.
# This script uses SFTTrainer from the 'trl' library, which simplifies
# the process of supervised fine-tuning and handles instruction-masking
# automatically.

# --- Prerequisites ---
# You need to install the necessary libraries first.
# pip install transformers datasets accelerate peft trl torch
# pip install deepseek-llm # or other DeepSeek-related libraries if needed

import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# --- Configuration ---
# Set the path to your data file and the model you want to fine-tune.
DATA_FILE = "data_train_processed.jsonl"
OUTPUT_DIR = "./deepseek-parlaylib-finetuned"
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base" # Using a base model for instruction tuning

# --- Step 1: Prepare your Dataset ---
# The script expects a JSONL file where each line is a JSON object
# with an "instruction" and a "response" key.
# This part of the code creates a dummy dataset for demonstration.
# In a real-world scenario, you would replace this with your actual data.
SYSTEM_INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a 
response that appropriately completes the request.

### Instruction:
{}\n{}

### Response:
{}
"""

def format_instruction_prompt(example):
    """
    Formats the instruction and response into the prompt string that
    the model will be trained on. This is the standard format for
    instruction tuning.
    """
    instruction = example["instruction"]
    input = example["input"]
    output = example["output"]
    return SYSTEM_INSTRUCTION_TEMPLATE.format(instruction, input, output)


# Load the dataset from the JSONL file
data = pd.read_json(DATA_FILE, lines=True)
dataset = Dataset.from_pandas(data)

# --- Step 2: Load the Model and Tokenizer ---
print(f"\nLoading model '{MODEL_NAME}' and its tokenizer...")
# We load the base model in 4-bit quantization to reduce VRAM usage
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
# DeepSeek's tokenizer handles chat formats and special tokens well
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Add a padding token if it doesn't exist, which is often needed for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Step 3: Configure and Run SFTTrainer ---
# We'll use PEFT (Parameter-Efficient Fine-Tuning) with LoRA.
# This technique only fine-tunes a small number of new parameters,
# which is much faster and less memory-intensive than full fine-tuning.
print("Setting up LoRA configuration for PEFT...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # Target common attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Set up the training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,  # Adjust based on your VRAM
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate a larger batch size
    warmup_steps=5,                 # Learning rate warmup
    max_steps=50,                   # Number of training steps (adjust for full dataset)
    learning_rate=2e-4,             # A good starting learning rate for LoRA
    logging_steps=10,               # Log progress every 10 steps
    report_to="none",               # No external logging for this example
    fp16=True,                      # Use mixed-precision training
)

# Use SFTTrainer from TRL for simplified fine-tuning
# This trainer automatically handles instruction-masking by only calculating
# the loss on the response portion of the formatted text.
print("Initializing the SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=format_instruction_prompt, # Our function to format the data
    # max_seq_length=512, # Max length of a single training example
)

# Fine-tune the model
print("\nStarting fine-tuning...")
trainer.train()

# Save the final model and tokenizer
print(f"\nFine-tuning complete. Saving model to '{OUTPUT_DIR}'...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model and tokenizer saved successfully.")
