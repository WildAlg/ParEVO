import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, DefaultDataCollator
from peft import PeftModel
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
from codebleu import calc_codebleu

import sys
sys.path.append("../..")
import re
import subprocess
import os
from finetune.utils import *
from utils import *

SYSTEM_PROMPT = (
    "You are an expert in C++ and the ParlayLib library."
    "{}"
    "{}"
    "Only write code that fits the description. Do not give other unnecessary code."
    "Just print the ParlayLib code and remove all comments and header files. "
    "Surround the generated ParlayLib code in #start and #end."
)

COMPLETE_CODE_SYSTEM_PROMPT = (
    "You are an expert in C++ and the ParlayLib library."
    "{}"
    "{}"
    "Write code that fits the description and place it between the markers in the given code." 
    "{}"
    "Do not write other unnecessary code."
    "Just print the ParlayLib code and remove all comments and header files. "
    "Surround the generated ParlayLib code in #start and #end."
)

HPC_SYSTEM_PROMPT = ("Below is an instruction that describes a task. Write a response that appropriately completes the request."
"### Instruction:"
"{}\n{}"
"Only write code that fits the description. Do not give other unnecessary code. For example, do not place the code in a `main` function."
"### Response:"
)

# A regex pattern to extract the code between the #start and #end markers
# CODE_PATTERN = re.compile(r"#start\s*([\s\S]*?)\s*#end", re.DOTALL)

def extract_parlaylib_code(text):
    """
    Extracts translated ParlayLib code from a model response.

    Supports:
    1. Between '#start' and '#end'
    2. Between ```cpp and ```
    """
    # Try #start ... #end first
    match = re.search(r"#start(.*?)#end", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ```cpp ... ```
    match = re.search(r"```cpp\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def load_model(model_path, is_lora=False):
    """
    Loads the model and tokenizer, handles LoRA and device mapping.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path if not is_lora else model_path.replace("-lora", ""))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_lora:
        base_model_path = model_path.replace("-lora", "")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
    
    model.eval()
    return model, tokenizer


def generate_completion(batch_prompts, model, tokenizer, device, args):
    """
    Generates completions for a batch of prompts.
    """
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and remove the prompt from the generated text
    decoded_batch = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    completions = []
    for prompt, decoded in zip(batch_prompts, decoded_batch):
        print("decoded: ", decoded)
        code = extract_parlaylib_code(decoded[len(prompt):].strip())
        completions.append(code)
    
    return completions


def insert_generated_code(eval_output_file: str, data_test_hollow_file: str, combined_output_file: str):
    """
    Combines two JSONL files:
    1. A file with model-generated code in a 'generated' field.
    2. A file with C++ test code containing markers for insertion.

    The function inserts the 'generated' code into the 'test_code' field of the 
    corresponding JSON object, then saves the new combined JSONL to an output file.
    """
    # Regex to find the markers and the empty space in between
    pattern = re.compile(r"(// begin of LLM generated code\n)(.*?)(\n // end of LLM generated code)", re.DOTALL)
    
    try:
        with open(eval_output_file, 'r', encoding='utf-8') as eval_f, \
             open(data_test_hollow_file, 'r', encoding='utf-8') as data_f, \
             open(combined_output_file, 'w', encoding='utf-8') as out_f:
            
            # Use zip to iterate over both files line by line simultaneously
            for eval_line, data_line in zip(eval_f, data_f):
                try:
                    eval_data = json.loads(eval_line)
                    data_data = json.loads(data_line)

                    # Get the generated code and the hollow test code
                    generated_code = eval_data.get('generated', '')
                    test_code = data_data.get('test_code', '')
                    
                    if generated_code and test_code:
                        # Insert the generated code into the test code using regex substitution
                        # \1 and \3 are backreferences to the markers
                        # The generated code is placed in between
                        modified_test_code = re.sub(
                            pattern,
                            r"\1" + generated_code + r"\3",
                            test_code
                        )
                        data_data['test_code'] = modified_test_code
                    
                    # Write the modified JSON object to the output file
                    json.dump(data_data, out_f, ensure_ascii=False)
                    out_f.write('\n')

                except json.JSONDecodeError as e:
                    print(f"Skipping a malformed JSON line. Error: {e}")
                    continue

    except FileNotFoundError as e:
        print(f"Error: A file was not found. Please check paths. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def run_tests(args):
    """
    Iterates through the combined JSONL file, compiles, and runs each C++ test.
    """
    passed_tests = 0
    failed_tests = 0
    failed_tests_list = []
    
    # Clean up previous output files
    if os.path.exists(args.output):
        os.remove(args.output)
    
    base = os.path.splitext(args.combined_input)[0]
    failed_tests_file = f"{base}_failed_tests_list.txt"
    if os.path.exists(failed_tests_file):
        os.remove(failed_tests_file)

    # Make a temporary directory for compilation
    subprocess.run(["mkdir", "-p", "tmp"], check=True)

    with open(args.combined_input, 'r') as f:
        for idx, line in enumerate(f):
            example = json.loads(line)
            if example is None: continue
            try:
                test_code = example['test_code']
                test_name = example['test_name']
            except KeyError as e:
                print(f"Skipping example {idx} due to missing key: {e}")
                continue

            # Write the test code to a temporary C++ file
            cpp_path = os.path.join("tmp", f"{test_name}.cpp")
            bin_path = os.path.join("tmp", f"{test_name}")
            with open(cpp_path, 'w') as f_cpp:
                f_cpp.write(test_code)

            # Compile the C++ file
            compile_command = f"g++ -std=c++17 -I../../../parlaylib/include {test_name}.cpp -o {test_name} -lpthread"
            res = subprocess.run(compile_command, shell=True, cwd="tmp", capture_output=True, text=True)
            
            if (res.returncode != 0):
                print(f"Compilation failed for {test_name}.cpp: {res.stderr}")
                failed_tests += 1
                failed_tests_list.append(test_name)
                with open(failed_tests_file, "a") as f_fail:
                    f_fail.write(f"{test_name}\t{idx}\n")
                if os.path.exists(cpp_path): os.remove(cpp_path)
                continue
            
            print(f"Compilation successful for {test_name}.cpp")
            
            # Run the compiled executable
            run_command = f"./{test_name}"
            res_run = subprocess.run(run_command, shell=True, cwd="tmp", capture_output=True, text=True)

            if (res_run.returncode != 0):
                print(f"Execution failed for {test_name}.cpp: {res_run.stderr}")
                failed_tests += 1
                failed_tests_list.append(test_name)
                with open(failed_tests_file, "a") as f_fail:
                    f_fail.write(f"{test_name}\t{idx}\n")
                if os.path.exists(cpp_path): os.remove(cpp_path)
                if os.path.exists(bin_path): os.remove(bin_path)
                continue
            
            print(f"Execution successful for {test_name}.cpp")
            passed_tests += 1

            # Append the successful result
            with open(args.output, "a") as fout:
                fout.write(json.dumps(example) + "\n")

            # Remove generated files
            if os.path.exists(cpp_path): os.remove(cpp_path)
            if os.path.exists(bin_path): os.remove(bin_path)

    total_tests = passed_tests + failed_tests
    print("============================================================\n")
    print(f"Total tests: {total_tests}, Passed: {passed_tests}({passed_tests/total_tests*100:.1f}%), Failed: {failed_tests}({failed_tests/total_tests*100:.1f}%)")
    print("Failed tests list: ", failed_tests_list)

def generate_metric_scores(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=args.cpu)
    model, tokenizer = load_model(args.model_path, is_lora=args.is_lora)
    
    # Load and preprocess the dataset
    dataset = load_dataset('json', data_files={'test': args.test_file}, split='test')
    print("dataset columns:", dataset.column_names)
    print("Dataset loaded with {} examples.".format(len(dataset)))

    def preprocess_function(examples):
        # Safely get columns, defaulting to an empty list if a key is missing
        instructions = examples.get("instruction", [None] * len(examples["instruction"]))
        inputs = examples.get("input", [None] * len(examples["input"]))
        outputs = examples.get("output", [None] * len(examples["output"]))
        hollow_code = examples.get("test_code_hollow", [None] * len(examples["test_code_hollow"]))
        ref_code = examples.get("test_code", [None] * len(examples["test_code"]))
        
        prompts = []
        references = []
        for instruction, input_text, output, hollow, ref in zip(instructions, inputs, outputs, hollow_code, ref_code):
            # Ensure all components are strings, handling None values gracefully
            instruction = str(instruction) if instruction is not None else ""
            input_text = str(input_text) if input_text is not None else ""
            output = str(output) if output is not None else ""
            hollow = str(hollow) if hollow is not None else ""
            ref = str(ref) if ref is not None else ""
            
            # Construct the prompt
            if args.hpc_coder:
                prompt = HPC_SYSTEM_PROMPT.format(instruction, input_text)
            else:
                if args.prompt_complete_code:
                    prompt = COMPLETE_CODE_SYSTEM_PROMPT.format(instruction, input_text, hollow)
                else:
                    prompt = SYSTEM_PROMPT.format(instruction, input_text)
                
            prompts.append(prompt)
            # Ensure reference is a string
            if args.prompt_complete_code:
                references.append(str(ref) if ref is not None else "")
            else:
                references.append(str(output) if output is not None else "")
        
        return {"prompt": prompts, "references": references}
    

    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["instruction", "input", "output"])
    
    # Create a DataLoader
    # valid_data_collator = DefaultDataCollator(return_tensors="pt")
    def collate_fn(batch):
        # batch is a list of dicts: [{"prompt": str, "references": str}, ...]
        prompts = [item["prompt"] for item in batch]
        references = [item["references"] for item in batch]
        return {"prompt": prompts, "references": references}
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Prepare with Accelerate
    model, dataloader = accelerator.prepare(model, dataloader)

    # Load metrics
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")
    rouge_metric = evaluate.load("rouge")

    output_path = Path(args.output_file)
    output_file = open(output_path, "w", encoding="utf-8")

    all_generated_codes = []
    all_references = []
    for batch in tqdm(dataloader):
        prompts = batch["prompt"]
        references = batch["references"]

        # Inference
        generated_codes = generate_completion(
            prompts,
            accelerator.unwrap_model(model),
            tokenizer,
            accelerator.device,
            args
        )

        # Save each example to JSONL
        filtered_preds = []
        filtered_refs = []
        for gen, ref in zip(generated_codes, references):
            json_line = {
                "generated": gen
            }
            output_file.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            if gen is not None:
                filtered_preds.append(gen)
                filtered_refs.append(ref)

        all_generated_codes.extend(filtered_preds)
        all_references.extend(filtered_refs)

        # Add predictions and references to metrics
        bleu_metric.add_batch(predictions=filtered_preds, references=filtered_refs)
        chrf_metric.add_batch(predictions=filtered_preds, references=filtered_refs)
        rouge_metric.add_batch(predictions=filtered_preds, references=filtered_refs)

    output_file.close()

    # Compute metrics
    bleu_res = bleu_metric.compute()
    chrf_res = chrf_metric.compute()
    rouge_res = rouge_metric.compute()
    codebleu_res = calc_codebleu(all_generated_codes, all_references, lang=args.language)

    if accelerator.is_main_process:
        if args.save_metric_scores:
            with open("metric_scores.txt", 'a') as f:
                f.write(f"\n--- Evaluation Results of {args.model_path} with `prompt_complete_code = {args.prompt_complete_code}` ---\n")
                f.write(f"BLEU: {bleu_res['score']:.2f}\n")
                f.write(f"ChrF: {chrf_res['score']:.2f}\n")
                f.write(f"ROUGE-L: {rouge_res['rougeL']:.4f}\n")
                f.write(f"CodeBLEU: {100 * codebleu_res['codebleu']:.4f}\n")
                f.write(f"More on CodeBLEU: \n")
                f.write(f"ngram_match_score: {100 * codebleu_res['ngram_match_score']:.4f}\n")
                f.write(f"weighted_ngram_match_score: {100 * codebleu_res['weighted_ngram_match_score']:.4f}\n")
                f.write(f"syntax_match_score: {100 * codebleu_res['syntax_match_score']:.4f}\n")
                f.write(f"dataflow_match_score: {100 * codebleu_res['dataflow_match_score']:.4f}\n")


if __name__ == "__main__":
    # Create the main parser
    parser = argparse.ArgumentParser(description="Evaluate model on JSONL with BLEU, ChrF, ROUGE-L, and CodeBLEU.")
    
    # Use subparsers to handle different modes (merge, shuffle, split)
    subparsers = parser.add_subparsers(dest='command', required=True, help='sub-command help')

    # Create the parser for the "metric" command
    metric_parser = subparsers.add_parser('metric', help='Generate metric scores.')
    metric_parser.add_argument("test_file", help="Path to JSONL test file.")
    metric_parser.add_argument("model_path", help="Model path (HF or LoRA).")
    metric_parser.add_argument("--prompt_complete_code", action="store_true", help="Whether the input prompt has the hollow code snippet.")
    metric_parser.add_argument("--language", default="cpp", help="Programming language for CodeBLEU.")
    metric_parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")
    metric_parser.add_argument("--max_new_tokens", default=1024, help="Maximum number of new tokens generated.")
    metric_parser.add_argument("--top_p", default=0.95, help="top p sampling.")
    metric_parser.add_argument("--is_lora", action="store_true", help="If the model is LoRA fine-tuned.")
    metric_parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    metric_parser.add_argument("--cpu", action="store_true", help="Force evaluation on CPU.")
    metric_parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    metric_parser.add_argument("--output_file", default="eval_output.jsonl", help="The generated code with the given input JSONL file.")
    metric_parser.add_argument("--save_metric_scores", action="store_true", help="Save the metric scores to `metric_scores.txt` file.")
    metric_parser.add_argument("--hpc_coder", action="store_true", help="If we are evaluating HPC-Coder.")
    
    # Create the parser for the "verify" command
    verify_parser = subparsers.add_parser('verify', help='Verify (compile & run) generated code.')
    verify_parser.add_argument("--eval_output", type=str, required=True, help="Input JSONL file with model generated code.")
    verify_parser.add_argument("--hollow_tests", type=str, required=True, help="Input JSONL file with hollow test cases.")
    verify_parser.add_argument("--output", type=str, required=True, help="Output JSONL file with test results.")

    args = parser.parse_args()


    if args.command == 'metric':
        generate_metric_scores(args)
    elif args.command == 'verify':
        # Step 1: Insert generated code into the hollow test file
        combined_file_path = "tmp/combined_tests.jsonl"
        print(f"Inserting generated code into hollow tests and saving to '{combined_file_path}'...")
        insert_generated_code(args.eval_output, args.hollow_tests, combined_file_path)
        print("Code insertion complete.")

        # Step 2: Run the tests with the newly created combined file
        run_args = argparse.Namespace(combined_input=combined_file_path, output=args.output)
        print(f"\nStarting test execution using the combined file...")
        run_tests(run_args)

