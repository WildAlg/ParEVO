#!/usr/bin/env python

"""
SFT (instruction tuning) for MoE architectures using TRL + LoRA.

Dataset format (JSONL):
  { "instruction": "...", "input": "...", "output": "..." }

This script tries to be MoE-safe by:
- targeting attention + FFN (experts) with LoRA
- avoiding router / gating modules by default
"""

import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# -----------------------------
# CONFIG
# -----------------------------

DATA_FILE = "data_train_processed.jsonl"

# EXAMPLE: replace this with the MoE model you want
# e.g. "deepseek-ai/deepseek-coder-v2-lite-base", "mistralai/Mixtral-8x7B-Instruct-v0.1", etc.
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

OUTPUT_DIR = "./moe-sft-out"

SYSTEM_INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a 
response that appropriately completes the request.

### Instruction:
{}\n{}

### Response:
{}
"""

def format_instruction_prompt(example):
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    return SYSTEM_INSTRUCTION_TEMPLATE.format(instruction, inp, output)


# -----------------------------
# LOAD DATASET
# -----------------------------

print(f"\nLoading dataset from {DATA_FILE} ...")
data = pd.read_json(DATA_FILE, lines=True)
dataset = Dataset.from_pandas(data)

# -----------------------------
# LOAD MODEL + TOKENIZER
# -----------------------------

print(f"\nLoading MoE model '{MODEL_NAME}' and tokenizer...")

trust_remote = True  # most MoE models with custom code need this; turn off if standard HF
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=trust_remote,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=trust_remote)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# LORA CONFIG (MOE-AWARE)
# -----------------------------

"""
For MoE models, we usually LoRA:
- attention projections: q_proj, k_proj, v_proj, o_proj
- FFN projections in experts: e.g. w1/w2/w3, gate_proj/up_proj/down_proj, etc.

We normally AVOID router/gate networks (often named 'gate', 'router', 'switch').
For each specific MoE model, inspect model.print() to get exact module names.
"""

# Common default for LLaMA/Mixtral-style MoE, override if needed:
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "w1", "w2", "w3",            # Mixtral experts
    "gate_proj", "up_proj", "down_proj",  # LLaMA-style FFNs
]

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

print("\nSetting up LoRA configuration for MoE...")
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=TARGET_MODULES,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

# -----------------------------
# TRAINING ARGS
# -----------------------------

print("Setting up TrainingArguments...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # MoE models are big; keep this small
    gradient_accumulation_steps=8,
    max_steps=500,                  # adjust as needed
    warmup_steps=50,
    learning_rate=2e-4,
    logging_steps=10,
    report_to="none",
    fp16=True,
    save_strategy="steps",
    save_steps=200,
)

# -----------------------------
# SFT TRAINER
# -----------------------------

print("Initializing SFTTrainer for MoE...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=format_instruction_prompt,
    max_seq_length=2048,   # adjust based on model context length
)

print("\nStarting fine-tuning MoE model...")
trainer.train()

print(f"\nFine-tuning complete. Saving model to '{OUTPUT_DIR}'...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")
