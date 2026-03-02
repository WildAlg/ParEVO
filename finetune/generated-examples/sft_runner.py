#!/usr/bin/env python

"""
Unified SFT runner for Code Llama + DeepSeek Coder using TRL's SFTTrainer + LoRA.
Now with YAML config support.

Dataset format (JSONL):
  {
    "instruction": "...",
    "input": "...",
    "output": "..."
  }
"""

import os
import argparse
import yaml
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# ------------------------------------------------------------
# 1) Model registry: Code Llama + DeepSeek Coder v1
# ------------------------------------------------------------

CODE_MODELS = {
    # ---------- Code Llama base ----------
    "codellama-7b-base": {
        "hf_name": "meta-llama/CodeLlama-7b-hf",
        "size_class": "7b",
        "max_seq_length": 4096,
        "lora_r": 8,
    },
    "codellama-13b-base": {
        "hf_name": "meta-llama/CodeLlama-13b-hf",
        "size_class": "13b",
        "max_seq_length": 4096,
        "lora_r": 8,
    },
    "codellama-34b-base": {
        "hf_name": "meta-llama/CodeLlama-34b-hf",
        "size_class": "34b",
        "max_seq_length": 4096,
        "lora_r": 16,
    },
    "codellama-70b-base": {
        "hf_name": "meta-llama/CodeLlama-70b-hf",
        "size_class": "70b",
        "max_seq_length": 4096,
        "lora_r": 16,
    },

    # ---------- Code Llama Instruct ----------
    "codellama-7b-instruct": {
        "hf_name": "meta-llama/CodeLlama-7b-Instruct-hf",
        "size_class": "7b",
        "max_seq_length": 4096,
        "lora_r": 8,
    },
    "codellama-13b-instruct": {
        "hf_name": "meta-llama/CodeLlama-13b-Instruct-hf",
        "size_class": "13b",
        "max_seq_length": 4096,
        "lora_r": 8,
    },
    "codellama-34b-instruct": {
        "hf_name": "meta-llama/CodeLlama-34b-Instruct-hf",
        "size_class": "34b",
        "max_seq_length": 4096,
        "lora_r": 16,
    },
    "codellama-70b-instruct": {
        "hf_name": "meta-llama/CodeLlama-70b-Instruct-hf",
        "size_class": "70b",
        "max_seq_length": 4096,
        "lora_r": 16,
    },

    # ---------- DeepSeek Coder base ----------
    "deepseek-1.3b-base": {
        "hf_name": "deepseek-ai/deepseek-coder-1.3b-base",
        "size_class": "1.3b",
        "max_seq_length": 16384,
        "lora_r": 8,
    },
    "deepseek-6.7b-base": {
        "hf_name": "deepseek-ai/deepseek-coder-6.7b-base",
        "size_class": "7b",
        "max_seq_length": 16384,
        "lora_r": 8,
    },
    "deepseek-7b-base-v1.5": {
        "hf_name": "deepseek-ai/deepseek-coder-7b-base-v1.5",
        "size_class": "7b",
        "max_seq_length": 16384,
        "lora_r": 8,
    },
    "deepseek-33b-base": {
        "hf_name": "deepseek-ai/deepseek-coder-33b-base",
        "size_class": "33b",
        "max_seq_length": 16384,
        "lora_r": 16,
    },

    # ---------- DeepSeek Coder instruct ----------
    "deepseek-1.3b-instruct": {
        "hf_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "size_class": "1.3b",
        "max_seq_length": 16384,
        "lora_r": 8,
    },
    "deepseek-6.7b-instruct": {
        "hf_name": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "size_class": "7b",
        "max_seq_length": 16384,
        "lora_r": 8,
    },
    "deepseek-7b-instruct-v1.5": {
        "hf_name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "size_class": "7b",
        "max_seq_length": 16384,
        "lora_r": 8,
    },
    "deepseek-33b-instruct": {
        "hf_name": "deepseek-ai/deepseek-coder-33b-instruct",
        "size_class": "33b",
        "max_seq_length": 16384,
        "lora_r": 16,
    },
}

# ------------------------------------------------------------
# 2) Prompt formatting
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 3) Simple heuristics for batch size by model size
# ------------------------------------------------------------

def default_batch_and_accum(size_class: str):
    """
    Very rough defaults; override via YAML if needed.
    """
    if size_class in ["1.3b"]:
        return 8, 1     # small models
    if size_class in ["7b"]:
        return 2, 4
    if size_class in ["13b"]:
        return 1, 8
    if size_class in ["33b", "34b"]:
        return 1, 16
    if size_class in ["70b"]:
        return 1, 32
    return 2, 4

# ------------------------------------------------------------
# 4) Main runner with YAML config
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT LoRA runner for Code Llama / DeepSeek Coder (YAML-driven).")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    # Load YAML
    with open(args.config, "r") as f:
        ycfg = yaml.safe_load(f)

    model_key = ycfg["model"]
    if model_key not in CODE_MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Valid keys: {list(CODE_MODELS.keys())}")

    cfg_model = CODE_MODELS[model_key]
    hf_name = cfg_model["hf_name"]
    size_class = cfg_model["size_class"]
    base_max_seq_length = cfg_model["max_seq_length"]
    base_lora_r = cfg_model["lora_r"]

    data_file = ycfg.get("data_file", "data_train_processed.jsonl")
    output_dir = ycfg.get("output_dir", f"./{model_key}-sft")

    # Training overrides
    training_cfg = ycfg.get("training", {}) or {}
    max_steps = int(training_cfg.get("max_steps", 1000))

    bs_default, ga_default = default_batch_and_accum(size_class)
    per_device_train_batch_size = int(training_cfg.get("per_device_train_batch_size", bs_default))
    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", ga_default))
    learning_rate = float(training_cfg.get("learning_rate", 2e-4))
    warmup_steps = int(training_cfg.get("warmup_steps", 50))
    logging_steps = int(training_cfg.get("logging_steps", 10))

    # LoRA overrides
    lora_cfg = ycfg.get("lora", {}) or {}
    lora_r = int(lora_cfg["r"]) if lora_cfg.get("r") is not None else base_lora_r
    alpha_multiplier = float(lora_cfg.get("alpha_multiplier", 2.0))
    lora_alpha = int(alpha_multiplier * lora_r)
    lora_dropout = float(lora_cfg.get("dropout", 0.05))

    # max_seq_length override
    max_seq_length = int(ycfg.get("max_seq_length", base_max_seq_length or 2048))

    print(f"\n==> Config loaded from {args.config}")
    print(f"    model_key: {model_key}")
    print(f"    HF model: {hf_name}")
    print(f"    size_class: {size_class}")
    print(f"    max_seq_length: {max_seq_length}")
    print(f"    data_file: {data_file}")
    print(f"    output_dir: {output_dir}")
    print(f"    LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    print(f"    max_steps: {max_steps}")
    print(f"    batch_size: {per_device_train_batch_size}, grad_accum: {gradient_accumulation_steps}")
    print(f"    learning_rate: {learning_rate}, warmup_steps: {warmup_steps}, logging_steps: {logging_steps}")

    print(f"\n==> Loading dataset from {data_file}")
    data = pd.read_json(data_file, lines=True)
    dataset = Dataset.from_pandas(data)

    trust_remote = hf_name.startswith("deepseek-ai/")

    print(f"\n==> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("==> Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        trust_remote_code=trust_remote,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("==> Setting up LoRA config...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("==> Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        report_to="none",
        fp16=True,
    )

    print("==> Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_instruction_prompt,
        max_seq_length=max_seq_length,
    )

    print("\n==> Starting training...")
    trainer.train()

    print(f"\n==> Saving model and tokenizer to {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("==> Done.")

if __name__ == "__main__":
    main()
