# Finetuning instructions

You may use `sft_runner.py` for finetuning (`instruction_ft.py` is the old script). By default, we use `deepseek-coder-6.7b-base`, but you can easily modify `MODEL_NAME` to perform LoRA finetuning on different base models. 

You can run `sft_runner.py` by:
```sh
# Example: DeepSeek 6.7B base
python sft_runner.py --model deepseek-6.7b-base --data_file data_train_processed.jsonl --max_steps 500

# Example: CodeLlama 7B Python
python sft_runner.py --model codellama-7b-base --data_file data_train_processed.jsonl --max_steps 1000

# Override batch size / grad-accum
python sft_runner.py --model codellama-13b-python --batch_size 1 --grad_accum_steps 8
```
Note that you have to request for access for `meta-llama` models on Hugging Face. 


To finetune models with Mixture-of-Expert structure, run 
```sh
python sft_moe_runner.py --config configs/deepseek-coder-v2-lite-base.yaml
```

## Set up your environment
1. Ensure hardware:
The test environment uses 1 `h200` (`vRAM = 141 GB`) and 120GB memory.

2. Load necessary modules and activate virtual environment 
```sh
module load miniconda
module load GCCcore/12.2.0
module load GCC/12.2.0   
module load CUDA/12.1.1
conda activate vllm-env
```
3. Pip install `pyyaml` (if YAML is used)
```sh
pip install pyyaml
```

## Use problem-sol dataset
```sh
SCRATCH_ROOT=path/to/scratch
mkdir -p "$SCRATCH_ROOT/hf_cache"
mkdir -p "$SCRATCH_ROOT/tmp"  
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_HUB_CACHE="$SCRATCH_ROOT/hf_cache/hub"
python sft_problem_sol.py --config configs/config_sft_qwen3_rust.yaml --scratch-dir $SCRATCH_ROOT
```


## Here are all Code Llama model names (HF)

From Meta’s official Code Llama Family collection 

### Base models
```sh
meta-llama/CodeLlama-7b-hf
meta-llama/CodeLlama-13b-hf
meta-llama/CodeLlama-34b-hf
meta-llama/CodeLlama-70b-hf
```

### Instruct
```sh
meta-llama/CodeLlama-7b-Instruct-hf
meta-llama/CodeLlama-13b-Instruct-hf
meta-llama/CodeLlama-34b-Instruct-hf
meta-llama/CodeLlama-70b-Instruct-hf
```

You can literally do:
```sh
MODEL_NAME = "meta-llama/CodeLlama-13b-Python-hf"
```
and keep the rest of the script the same (with the LoRA config I’ll give below).

## All DeepSeek Coder v1 model names
From DeepSeek’s DeepSeek-Coder collection 

### Instruct
```sh
deepseek-ai/deepseek-coder-33b-instruct
deepseek-ai/deepseek-coder-6.7b-instruct
deepseek-ai/deepseek-coder-7b-instruct-v1.5
deepseek-ai/deepseek-coder-1.3b-instruct
```

### Base
```sh
deepseek-ai/deepseek-coder-33b-base
deepseek-ai/deepseek-coder-6.7b-base
deepseek-ai/deepseek-coder-7b-base-v1.5
deepseek-ai/deepseek-coder-1.3b-base
```

(DeepSeek Coder V2 is a separate MoE family; you can adapt the script.)


## One **unified config dict** in the script

This gives you:

* model name
* recommended LoRA rank
* suggested max seq length
* a “size class” so you can adjust batch size

```python
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
```

Then at the top of the script we have:

```python
MODEL_KEY = "deepseek-6.7b-instruct"  # or any of the keys above
cfg = CODE_MODELS[MODEL_KEY]
MODEL_NAME = cfg["hf_name"]
MAX_SEQ_LENGTH = cfg["max_seq_length"]
LORA_R = cfg["lora_r"]
```

and plug `LORA_R` into `LoraConfig`.

---

## 4. Shared **LoRA config** for all of them

For **Code Llama** and **DeepSeek Coder v1**, the module names are LLaMA-style, so you can use the same `target_modules`:

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=2 * LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

---

## 5. Rough VRAM / batch size guidelines (FP16 + LoRA)

Very rough ballpark (per **single A100-class 80GB**; scale down for smaller GPUs):

* **1.3B–2B**

  * VRAM: 4–8 GB
  * `per_device_train_batch_size`: 8–16
  * `lora_r = 8` is plenty.

* **7B-ish (CodeLlama 7B, DeepSeek 6.7/7B)**

  * VRAM: ~16–24 GB for full model in FP16; with LoRA + 4-bit you can squeeze on 12–16 GB
  * Start with:

    ```python
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 4
    ```
  * `lora_r = 8`.

* **13B**

  * VRAM: 32 GB+ comfortable in FP16; 24 GB OK with 4-bit + LoRA
  * Start with: `batch_size=1–2`, `grad_accum_steps=4–8`
  * `lora_r = 8`.

* **33B / 34B**

  * VRAM: basically multi-GPU or 80GB + 4-bit/QLoRA territory
  * Start with: `batch_size=1`, `grad_accum_steps=8+`
  * `lora_r = 16`.

* **70B**

  * Needs tensor/ZeRO sharding or multi-GPU; same LoRA pattern, just very small batch.


