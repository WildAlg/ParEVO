#!/usr/bin/env python

"""
Universal MoE SFT runner using TRL's SFTTrainer + LoRA.

Supports:
  - Mixtral-8x7B (Instruct/base)
  - DeepSeek-Coder-V2 / V2-Lite (Base/Instruct)
  - Any custom MoE model via YAML config (model_key: custom)

Dataset format (JSONL):
  { "instruction": "...", "input": "...", "output": "..." }
"""

import argparse
import yaml
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# ------------------------------------------------------------
# 1) MoE model registry
# ------------------------------------------------------------

MOE_MODELS = {
    # ---------- Mixtral ----------
    # FFN experts: w1, w2, w3
    # Attention: q_proj, k_proj, v_proj, o_proj
    "mixtral-8x7b": {
        "hf_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "size_class": "8x7b",
        "max_seq_length": 4096,
        "trust_remote_code": True,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "w1", "w2", "w3",
        ],
    },
    "mixtral-8x7b-base": {
        "hf_name": "mistralai/Mixtral-8x7B-v0.1",
        "size_class": "8x7b",
        "max_seq_length": 4096,
        "trust_remote_code": True,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "w1", "w2", "w3",
        ],
    },

    # ---------- DeepSeek-Coder-V2-Lite ----------
    # Attention: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
    # MoE FFN experts: w1, w2, w3
    "deepseek-coder-v2-lite-base": {
        "hf_name": "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        "size_class": "16b-moe-lite",
        "max_seq_length": 16384,
        "trust_remote_code": True,
        "target_modules": [
            "q_a_proj", "q_b_proj",
            "kv_a_proj_with_mqa", "kv_b_proj",
            "o_proj",
            "w1", "w2", "w3",
        ],
    },
    "deepseek-coder-v2-lite-instruct": {
        "hf_name": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "size_class": "16b-moe-lite",
        "max_seq_length": 16384,
        "trust_remote_code": True,
        "target_modules": [
            "q_a_proj", "q_b_proj",
            "kv_a_proj_with_mqa", "kv_b_proj",
            "o_proj",
            "w1", "w2", "w3",
        ],
    },

    # ---------- DeepSeek-Coder-V2 full ----------
    "deepseek-coder-v2-base": {
        "hf_name": "deepseek-ai/DeepSeek-Coder-V2-Base",
        "size_class": "big-moe",
        "max_seq_length": 16384,
        "trust_remote_code": True,
        "target_modules": [
            "q_a_proj", "q_b_proj",
            "kv_a_proj_with_mqa", "kv_b_proj",
            "o_proj",
            "w1", "w2", "w3",
        ],
    },
    "deepseek-coder-v2-instruct": {
        "hf_name": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "size_class": "big-moe",
        "max_seq_length": 16384,
        "trust_remote_code": True,
        "target_modules": [
            "q_a_proj", "q_b_proj",
            "kv_a_proj_with_mqa", "kv_b_proj",
            "o_proj",
            "w1", "w2", "w3",
        ],
    },

    # ---------- Custom ----------
    # Use model_key: custom in YAML and specify hf_name, target_modules, etc.
    "custom": {
        # everything comes from YAML
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
# 3) Batch-size heuristics for big MoEs
# ------------------------------------------------------------

def default_batch_and_accum(size_class: str):
    """
    Very rough defaults; override via YAML if needed.
    """
    if size_class in ["8x7b", "16b-moe-lite"]:
        return 1, 8
    if size_class in ["big-moe"]:
        return 1, 16
    # generic fallback
    return 1, 8

# ------------------------------------------------------------
# 4) Main runner
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Universal MoE SFT + LoRA runner (Mixtral, DeepSeek-Coder-V2, custom MoE)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        ycfg = yaml.safe_load(f)

    model_key = ycfg["model_key"]
    if model_key not in MOE_MODELS:
        raise ValueError(f"Unknown model_key '{model_key}'. Valid keys: {list(MOE_MODELS.keys())}")

    reg = MOE_MODELS[model_key]

    # HF model name
    hf_name = ycfg.get("hf_name", reg.get("hf_name"))
    if hf_name is None:
        raise ValueError("You must specify 'hf_name' in YAML for model_key='custom'.")

    size_class = ycfg.get("size_class", reg.get("size_class", "unknown"))
    trust_remote_code = bool(ycfg.get("trust_remote_code", reg.get("trust_remote_code", True)))
    base_max_seq_length = reg.get("max_seq_length", 2048)

    data_file = ycfg.get("data_file", "data_train_processed.jsonl")
    output_dir = ycfg.get("output_dir", f"./{model_key}-moe-sft")

    # Training section
    training_cfg = ycfg.get("training", {}) or {}
    max_steps = int(training_cfg.get("max_steps", 1000))
    bs_default, ga_default = default_batch_and_accum(size_class)
    per_device_train_batch_size = int(training_cfg.get("per_device_train_batch_size", bs_default))
    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", ga_default))
    learning_rate = float(training_cfg.get("learning_rate", 2e-4))
    warmup_steps = int(training_cfg.get("warmup_steps", 50))
    logging_steps = int(training_cfg.get("logging_steps", 10))

    # LoRA section
    lora_cfg = ycfg.get("lora", {}) or {}
    lora_r = int(lora_cfg.get("r", 8))
    alpha_multiplier = float(lora_cfg.get("alpha_multiplier", 2.0))
    lora_alpha = int(alpha_multiplier * lora_r)
    lora_dropout = float(lora_cfg.get("dropout", 0.05))

    # target_modules: from YAML override, else from registry
    tm_yaml = lora_cfg.get("target_modules")
    if tm_yaml is not None:
        target_modules = list(tm_yaml)
    else:
        target_modules = reg.get("target_modules")
        if not target_modules:
            raise ValueError(
                "No target_modules provided. For custom MoE, specify lora.target_modules in YAML."
            )

    # max_seq_length override
    max_seq_length = int(ycfg.get("max_seq_length", base_max_seq_length))

    print(f"\n==> Config: {args.config}")
    print(f"    model_key: {model_key}")
    print(f"    hf_name: {hf_name}")
    print(f"    size_class: {size_class}")
    print(f"    trust_remote_code: {trust_remote_code}")
    print(f"    max_seq_length: {max_seq_length}")
    print(f"    data_file: {data_file}")
    print(f"    output_dir: {output_dir}")
    print(f"    max_steps: {max_steps}")
    print(f"    batch_size: {per_device_train_batch_size}, grad_accum: {gradient_accumulation_steps}")
    print(f"    learning_rate: {learning_rate}, warmup_steps: {warmup_steps}, logging_steps: {logging_steps}")
    print(f"    LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    print(f"    target_modules: {target_modules}")

    # -----------------------------
    # Load dataset
    # -----------------------------
    print(f"\n==> Loading dataset from {data_file}")
    data = pd.read_json(data_file, lines=True)
    dataset = Dataset.from_pandas(data)

    # -----------------------------
    # Load model + tokenizer
    # -----------------------------
    print(f"\n==> Loading tokenizer for {hf_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("==> Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # -----------------------------
    # LoRA config
    # -----------------------------
    print("==> Setting up LoRA config ...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -----------------------------
    # TrainingArguments
    # -----------------------------
    print("==> Setting up TrainingArguments ...")
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
        save_strategy="steps",
        save_steps=max(200, logging_steps * 5),
    )

    # -----------------------------
    # SFTTrainer
    # -----------------------------
    print("==> Initializing SFTTrainer ...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_instruction_prompt,
        max_length=max_seq_length
    )

    print("\n==> Starting training ...")
    trainer.train()

    print(f"\n==> Saving model + tokenizer to {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("==> Done.")

if __name__ == "__main__":
    main()
