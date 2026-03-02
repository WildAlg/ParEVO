import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from copy import deepcopy

# --- Keywords for ParlayLib (used for weighted dropout) ---
parlaylib_keywords = {
    # Namespace
    "parlay",
    
    # Core Algorithms
    "parallel_for", "sort", "reduce", "scan", "filter", "map", "pack",
    "unique", "histogram", "count", "merge", "find", "copy", "equal", "tabulate",

    # Data Structures & Types
    "sequence", "slice", "delayed_sequence",
    
    # Utilities
    "make_slice", "get_num_workers", "set_num_workers", "random",
}

# --- Class to Programmatically Corrupt Code ---
class CodeCorruptor:
    """
    Applies the same DAE noise functions from training to a clean string of code.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        
        self.keyword_ids = {
            self.tokenizer.encode(k, add_special_tokens=False)[0] 
            for k in parlaylib_keywords if len(self.tokenizer.encode(k, add_special_tokens=False)) == 1
        }
        
        # Noise probabilities
        self.WORD_MASK_PROB = 0.15
        self.WORD_DROPOUT_PROB = 0.10
        self.WORD_SHUFFLE_PROB = 0.10
        self.WORD_REPLACEMENT_PROB = 0.05
        self.WORD_INSERTION_PROB = 0.05

    def corrupt(self, clean_code_text):
        """Takes a clean string and returns a corrupted version."""
        tokenized_input = self.tokenizer(clean_code_text, return_tensors=None)
        
        # The noise functions expect a list of features, so we wrap it
        feature = [{"input_ids": tokenized_input["input_ids"]}]
        corrupted_feature = self._add_noise(feature)
        
        # Decode the corrupted token IDs back into a string
        return self.tokenizer.decode(corrupted_feature[0]["input_ids"], skip_special_tokens=True)

    def _add_noise(self, features):
        """Applies a sequence of noise functions to the input features."""
        for feature in features:
            input_ids = feature["input_ids"]
            input_ids_np = np.array(input_ids, dtype=np.int64)

            input_ids_np = self._word_dropout(input_ids_np)
            input_ids_np = self._word_shuffle(input_ids_np)
            input_ids_np = self._word_replacement(input_ids_np)
            input_ids_np = self._word_insertion(input_ids_np)
            input_ids_np = self._word_mask(input_ids_np)
            
            feature["input_ids"] = input_ids_np.tolist()[:len(input_ids)]
        return features

    def _word_mask(self, x):
        num_to_mask = int(len(x) * self.WORD_MASK_PROB)
        if num_to_mask == 0: return x
        indices_to_mask = np.random.choice(len(x), num_to_mask, replace=False)
        x[indices_to_mask] = self.mask_token_id
        return x

    def _word_dropout(self, x):
        num_to_drop = int(len(x) * self.WORD_DROPOUT_PROB)
        if num_to_drop == 0: return x
        is_keyword = np.isin(x, list(self.keyword_ids))
        probs = np.ones(len(x)); probs[is_keyword] = 2.0
        probs[x == self.pad_token_id] = 0
        if np.sum(probs) == 0: return x
        probs /= np.sum(probs)
        indices_to_drop = np.random.choice(len(x), num_to_drop, replace=False, p=probs)
        return np.delete(x, indices_to_drop)

    def _word_shuffle(self, x):
        if np.random.rand() > self.WORD_SHUFFLE_PROB or len(x) < 3: return x
        start = np.random.randint(0, len(x) - 2)
        end = np.random.randint(start + 1, len(x))
        sub_sequence = x[start:end]; np.random.shuffle(sub_sequence)
        x[start:end] = sub_sequence
        return x
        
    def _word_replacement(self, x):
        num_to_replace = int(len(x) * self.WORD_REPLACEMENT_PROB)
        if num_to_replace == 0: return x
        indices_to_replace = np.random.choice(len(x), num_to_replace, replace=False)
        random_tokens = np.random.randint(0, self.tokenizer.vocab_size, num_to_replace)
        x[indices_to_replace] = random_tokens
        return x

    def _word_insertion(self, x):
        num_to_insert = int(len(x) * self.WORD_INSERTION_PROB)
        if num_to_insert == 0: return x
        indices_to_insert = np.random.choice(len(x), num_to_insert, replace=False)
        random_tokens = np.random.randint(0, self.tokenizer.vocab_size, num_to_insert)
        return np.insert(x, indices_to_insert, random_tokens)

def run_denoising_inference(base_model_path, lora_adapter_path, corrupted_code_snippet):
    """
    Loads a base model, applies a LoRA adapter, and runs a denoising inference example.

    Args:
        base_model_path (str): The identifier of the base model on the Hugging Face Hub.
        lora_adapter_path (str): The local path to the trained LoRA adapter directory.
        corrupted_code_snippet (str): The corrupted code string to be repaired by the model.
    """
    print("--- Denoising Inference Example ---")

    # 1. Load the base model and tokenizer
    print(f"Step 1: Loading base model from '{base_model_path}'...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
        device_map="auto",          # Automatically use the GPU if available
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 2. Add the custom mask token and resize model embeddings
    # This is a crucial step to ensure the tokenizer and model vocabularies match.
    print("Step 2: Adding '[MASK]' token and resizing model embeddings...")
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    model.resize_token_embeddings(len(tokenizer))

    # 3. Load the LoRA adapter and merge it into the base model
    print(f"Step 3: Loading LoRA adapter from '{lora_adapter_path}'...")
    # This loads the adapter weights and applies them to the base model
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    print("Step 4: Merging LoRA weights for faster inference...")
    # This combines the adapter weights with the base model's weights.
    # After this, the model behaves like a standard, fully-fine-tuned model.
    model = model.merge_and_unload()
    
    # Set the model to evaluation mode
    model.eval()

    # 5. Prepare the input for the model
    print("\n--- Input Code (Corrupted) ---")
    print(corrupted_code_snippet)
    
    inputs = tokenizer(corrupted_code_snippet, return_tensors="pt").to(model.device)

    # 6. Generate the reconstructed code
    print("\nGenerating reconstruction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024, # Give it enough space to generate the full function
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False, # Use greedy decoding for a deterministic reconstruction
        )
    
    reconstructed_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 7. Print the result
    print("\n--- Output Code (Reconstructed) ---")
    print(reconstructed_code.strip())


if __name__ == "__main__":
    # --- Configuration ---
    # Path to the original, pre-trained model
    BASE_MODEL_PATH = "deepseek-ai/deepseek-coder-6.7b-base"
    
    # Path to your saved LoRA adapter.
    # IMPORTANT: Update this path to where your 'final_adapter' directory is located.
    ADAPTER_PATH = "./deepseek-coder-6.7b-base-lora-dae/final_adapter"

    # A clean code snippet that we will programmatically corrupt
    clean_code_to_corrupt = """
    auto seqs = parlay::tabulate(25, [](size_t i) {
        return parlay::tabulate(4000, [i](size_t j) {
                return static_cast<int>(i * j);
                });
        });
    auto seq = parlay::flatten(std::move(seqs));
    auto answer = parlay::tabulate(100000, [](size_t k) {
        size_t i = k / 4000;
        size_t j = k % 4000;
        return static_cast<int>(i * j);
    });
"""

    # --- Programmatically generate the corrupted code ---
    print("Step 0: Generating corrupted code for the inference example...")
    # We need a tokenizer instance to initialize the corruptor
    temp_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    temp_tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    corruptor = CodeCorruptor(tokenizer=temp_tokenizer)
    corrupted_code_to_fix = corruptor.corrupt(clean_code_to_corrupt)

    # Run the inference function with the newly generated corrupted code
    run_denoising_inference(
        base_model_path=BASE_MODEL_PATH,
        lora_adapter_path=ADAPTER_PATH,
        corrupted_code_snippet=corrupted_code_to_fix
    )

