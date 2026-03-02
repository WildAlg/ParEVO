# This script generates a supervised fine-tuning dataset from OpenMP
# code by translating it to ParlayLib code using the Google GenerativeAI SDK.

import sys
sys.path.append("..")
import re
import json
from pathlib import Path
import time
import argparse
import google.generativeai as genai
import os
import code_bert_score
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm
import warnings

# Register tqdm with pandas to use progress_apply
tqdm.pandas()

# Initialize pandarallel for multi-core processing
# You can specify the number of workers (e.g., nb_workers=4)
# If not specified, it will use all available cores.
# pandarallel.initialize(progress_bar=True)

# A regex pattern to extract the code between the #start and #end markers
CODE_PATTERN = re.compile(r"#start\s*([\s\S]*?)\s*#end", re.DOTALL)

def extract_parlaylib_code(text):
    """
    Extracts the translated ParlayLib code from a model response.
    
    The code is expected to be between the markers '#start' and '#end'.
    """
    match = CODE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def load_dataset(filename):
    """Loads a JSONL file into a list of dictionaries."""
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

# Define the system prompt as a constant at the module level
SYSTEM_PROMPT = (
    "You are an expert in translating OpenMP code to ParlayLib code."
    "Given the C++ program below, translate it to use the ParlayLib libraries. "
    "Ensure that the ParlayLib program is compatible with the C++ program "
    "and preserves the semantics of the original code."
    "Just print the ParlayLib code and remove any unnecessary comments. "
    "Surround the generated ParlayLib code in #start and #end."
    "Include necessary header files such as <parlay/primitives.h> and <parlay/sequence.h>."
    f"\n### OpenMP Code:\n"
)

# def ask_gemini_for_translation(data_point, cacheed_model, args):
#     """
#     Sends a prompt to the Gemini API to translate OpenMP code to ParlayLib.
#     """
#     prompt = (
#         f"\n### OpenMP Code:\n{data_point}"
#         f"\n### ParLayLib Version: "
#     )
    
#     generation_config = {
#         "temperature": args.temperature,
#         "top_p": args.top_p,
#         "top_k": args.top_k,
#         "candidate_count": args.n_samples,
#     }

#     # model = genai.GenerativeModel(args.model)
#     # response = model.generate_content(prompt, generation_config=generation_config)
#     response = cached_model.generate_content(prompt, generation_config=generation_config)
#     return response


def ask_gemini_for_translation(data_point, model, args):
    """
    Sends a prompt to the Gemini API to translate OpenMP code to ParlayLib.
    """
    # The prompt is constructed to be as clear as possible.
    prompt = f"{SYSTEM_PROMPT}{data_point}"

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "candidate_count": args.n_samples,
    }

    response = model.generate_content(prompt, generation_config=generation_config)
    return response

def translate_main(args):
    """Main function to load data, ask Gemini, and save the results."""
    genai.configure(api_key=args.api_key)
    # cache = genai.caching.CachedContent.create(
    #     model=args.model,
    #     system_instruction=(
    #         "You are an expert in translating OpenMP code to ParlayLib code."
    #         "Given the C++ program below, translate it to use the ParlayLib libraries. "
    #         "Ensure that the ParlayLib program is compatible with the C++ program "
    #         "and preserves the semantics of the original code."
    #         "Just print the ParlayLib code and remove any unnecessary comments. "
    #         "Surround the generated ParlayLib code in #start and #end."
    #         "Include necessary header files such as <parlay/primitives.h> and <parlay/sequence.h>."
    #     )
    # )

    # # Now, create a new model instance that uses the cache.
    # cached_model = genai.GenerativeModel.from_cached_content(cached_content=cache)
    model = genai.GenerativeModel(args.model)

    dataset = load_dataset(args.input)
    
    # # Check if the output file already exists to resume from where it left off
    output_path = Path(args.output)
    processed_indices = set()
    # if output_path.exists():
    #     with open(output_path, 'r') as f:
    #         for line in f:
    #             item = json.loads(line)
    #             # Assuming omp_code is unique enough to identify a processed item
    #             processed_indices.add(item['omp_code'])

    # Open the output file in append mode.
    # This ensures new items are added without overwriting existing ones.
    with open(args.output, 'a') as f:
        for idx, data_point in enumerate(dataset):
            if idx < 5383: continue
            # Skip this data point if it's already been processed
            if data_point['code'] in processed_indices:
                print(f"Skipping already processed data point {idx+1}/{len(dataset)}...")
                continue
            
            print(f"Processing data point {idx+1}/{len(dataset)}...")
            
            try:
                response = ask_gemini_for_translation(data_point['code'], model, args)
                
                # Iterate through all generated candidates
                for candidate in response.candidates:
                    generated_text = candidate.content.parts[0].text
                    generated_code = extract_parlaylib_code(generated_text)
                    
                    item_to_save = {
                        "omp_code": data_point['code'],
                        "parlay_code": generated_code if generated_code else generated_text
                    }
                    
                    # Write the new item to the file immediately
                    f.write(json.dumps(item_to_save) + '\n')
                
                time.sleep(1) # A simple way to avoid overwhelming the API
                
            except Exception as e:
                print(f"Error processing data point {idx}: {e}")
                # Log the error and continue to the next data point
                continue
    
    # cache.delete()

# ================= process translated parlay code ============


def clean_code_snippet(code):
    """Removes markdown code block syntax from the snippet."""
    if code.strip().startswith("```cpp"):
        return code.replace("```cpp", "").replace("```", "").strip()
    return code.strip()

def calculate_score(reference, candidate):
    """Calculates the CodeBERTScore for C++ snippets."""
    # CodeBERTScore requires the references and candidates to be lists
    # The 'lang' parameter is set to 'cpp' for C++
    return code_bert_score.score(
        [clean_code_snippet(reference)], 
        [clean_code_snippet(candidate)], 
        lang="cpp"
    )[2].item()

def process_translated_data_points(input_file, filtered_output_file, checkpoint_interval=500):
    """
    Processes JSONL data in chunks, calculates scores,
    saves periodic checkpoints, and supports resume.
    """
    print("--- Starting CodeBERTScore Analysis with checkpoints & resume ---")

    checkpoint_path = input_file + ".checkpoint"

    # Load full dataset
    try:
        df_full = pd.read_json(input_file, lines=True)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return

    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        df_checkpoint = pd.read_json(checkpoint_path, lines=True)
        processed_rows = len(df_checkpoint)
        results = [df_checkpoint]
    else:
        print("No checkpoint found, starting fresh...")
        processed_rows = 0
        results = []

    # Process remaining chunks
    for start in range(processed_rows, len(df_full), checkpoint_interval):
        end = min(start + checkpoint_interval, len(df_full))
        chunk = df_full.iloc[start:end].copy()

        # similarity2omp
        chunk['similarity2omp'] = chunk.apply(
            lambda row: calculate_score(row['omp_code'], row['parlay_code']),
            axis=1
        )

        results.append(chunk)

        # Save checkpoint
        checkpoint_df = pd.concat(results, ignore_index=True)
        checkpoint_df.to_json(checkpoint_path, orient='records', lines=True)
        print(f"Checkpoint saved after {end} rows -> {checkpoint_path}")

    # Merge all processed chunks
    df = pd.concat(results, ignore_index=True)

    # Best matches
    best_match_indices = df.groupby('omp_code')['similarity2omp'].idxmax()
    best_matches_df = df.loc[best_match_indices]
    best_parlay_codes = best_matches_df.set_index('omp_code')['parlay_code']
    df['best_parlay_code'] = df['omp_code'].map(best_parlay_codes)

    # similarity2parlay
    df['similarity2parlay'] = df.apply(
        lambda row: 1.0 if row['parlay_code'] == row['best_parlay_code']
        else calculate_score(row['best_parlay_code'], row['parlay_code']),
        axis=1
    )

    df = df.drop(columns=['best_parlay_code'])

    # Final save
    df.to_json(input_file, orient='records', lines=True)
    best_matches_df.to_json(filtered_output_file, orient='records', lines=True)

    # Remove checkpoint (optional: keep if you want logs)
    os.remove(checkpoint_path)

    print("\n--- Analysis Complete! ---")
    print(f"Updated '{input_file}' with score attributes.")
    print(f"Filtered results saved to '{filtered_output_file}'.")


# ====================== Create Driver Code =======================

def create_driver_code_data_points(
    input_file: str, 
    output_file: str, 
    template: str, 
    model_name: str, 
    api_key: str, 
    checkpoint_interval: int = 50
):
    """
    Loads a JSONL file, reads ParlayLib code solutions, and generates
    a C++ driver code for each solution using a cached LLM API. The
    template and system instructions are cached to optimize performance.

    Args:
        input_file (str): Path to the input JSONL file containing ParlayLib solutions.
        output_file (str): Path to the output JSONL file to save the results.
        template (str): The C++ driver code template. This will be cached.
        model_name (str): The name of the model to use (e.g., 'gemini-2.5-flash-preview-05-20').
        api_key (str): Your API key for the Gemini API.
        checkpoint_interval (int): Number of data points to process before
                                   writing to the output file as a checkpoint.
    """
    
    # Configure the genai library with the provided API key
    genai.configure(api_key=api_key)

    # Define the system instruction for the LLM
    system_instruction = (
        "You are an expert C++ and parallel programming assistant. "
        "Your task is to complete the provided C++ driver code template for a given ParlayLib snippet. "
        "The template has TODO sections for you to fill in:"
        "1.  Variable and Data Structure Initialization: Initialize any variables or structs needed for the code snippet to compile and run."
        "2.  Paste the ParlayLib code: Place the provided code snippet in this section."
        "3.  Assertion and Verification: Write logic to verify the correctness of the code's output. You may need to create a simple, single-threaded reference implementation for comparison."
        "Ensure the final code is ready to compile and run. Do not add any extra explanations, "
        "just the code block."
    )

    

    # Create the cache for the system instruction and the driver template
    # This will be used as a "prefix" to every prompt to save tokens and time.
    cache = genai.caching.CachedContent.create(
        model=model_name,
        system_instruction=system_instruction,
        contents=[
            {"role": "user", "parts": [
                {"text": "C++ Driver Template:\n```cpp\n" + template + "\n```"}
            ]}
        ]
    )

    # Create a new model instance that uses the cache
    cached_model = genai.GenerativeModel.from_cached_content(cached_content=cache)

    # Check if output file exists and resume from a previous run
    processed_count = 0
    if os.path.exists(output_file):
        print(f"Resuming from existing file: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                processed_count += 1
        print(f"Skipping {processed_count} already processed data points.")

    # Load all input data points
    with open(input_file, 'r') as f:
        data_points = [json.loads(line) for line in f]

    # Process each data point, starting from the checkpoint
    new_data = []
    with open(output_file, 'a') as f:
        for idx, item in tqdm(enumerate(data_points), total=len(data_points), desc="Creating driver code"):
            # If we've already processed this data point, skip it
            if idx < processed_count:
                continue

            if 'parlay_code' in item and isinstance(item['parlay_code'], str):
                solution_code = item['parlay_code']
                
                # The prompt now only contains the user's code snippet.
                prompt = f"""
ParlayLib Code Snippet to Test:
```cpp
{solution_code}
```
"""
                
                # --- API Call to Gemini with Exponential Backoff ---
                retries = 0
                max_retries = 3
                response_text = ""
                
                while retries < max_retries:
                    try:
                        # Use the cached model for the generation
                        response = cached_model.generate_content(prompt)
                        prompt_tokens = response.usage_metadata.prompt_token_count
                        output_tokens = response.usage_metadata.candidates_token_count
                        total_tokens = response.usage_metadata.total_token_count
                        print(f"total_tokens: {total_tokens}")
                        response_text = response.candidates[0].content.parts[0].text
                        sleep(5)
                        break  # Success, exit the loop
                    except Exception as e:
                        retries += 1
                        wait_time = 2 ** retries  # Exponential backoff
                        tqdm.write(f"API call failed: {e}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                
                # If all retries fail, use a placeholder error message
                if not response_text:
                    response_text = "// Error: Failed to generate driver code after multiple retries."

                item['driver_code'] = response_text
                new_data.append(item)

            # Checkpoint the data
            if len(new_data) >= checkpoint_interval:
                for d in new_data:
                    f.write(json.dumps(d) + '\n')
                new_data = []

        # Write any remaining data points
        for d in new_data:
            f.write(json.dumps(d) + '\n')
    
    print("Driver code generation complete.")


# def create_driver_code_data_points(input_file, output_file, template, checkpiont_interval=500):
#     """
#     Loads a JSONL file, reads ParlayLib code solutions, and generates
#     a C++ driver code for each solution using a large language model API.
#     The updated data points, including the new 'driver_code' attribute,
#     are saved to a new JSONL file.

#     Args:
#         input_file (str): Path to the input JSONL file containing ParlayLib solutions.
#         output_file (str): Path to the output JSONL file to save the results.
#         template (str): The C++ driver code template to be completed by the LLM.
#         checkpoint_interval (int): Number of data points to process before
#                                    writing to the output file as a checkpoint.
#     """
    
#     # Check if output file exists and resume from a previous run
#     processed_count = 0
#     if os.path.exists(output_file):
#         print(f"Resuming from existing file: {output_file}")
#         with open(output_file, 'r') as f:
#             for line in f:
#                 processed_count += 1
#         print(f"Skipping {processed_count} already processed data points.")

#     # Load all input data points
#     with open(input_file, 'r') as f:
#         data_points = [json.loads(line) for line in f]

#     # Process each data point, starting from the checkpoint
#     new_data = []
#     with open(output_file, 'a') as f:
#         for idx, item in tqdm(enumerate(data_points), total=len(data_points), desc="Creating driver code"):
#             # If we've already processed this data point, skip it
#             if idx < processed_count:
#                 continue

#             # We now look for 'parlay_code'
#             if 'parlay_code' in item and isinstance(item['parlay_code'], str):
#                 solution_code = item['parlay_code']
                
#                 # Construct the prompt for the LLM
#                 prompt = CREATE_DRIVER_CODE_SYSTEM_TEMPLATE.format(template, solution_code)
                
#                 # --- API Call to Gemini with Exponential Backoff ---
#                 payload = {
#                     "contents": [{"parts": [{"text": prompt}]}],
#                 }
#                 apiKey = "" # Use empty string for Canvas
#                 apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=" + apiKey

#                 retries = 0
#                 max_retries = 5
#                 response_text = ""
                
#                 while retries < max_retries:
#                     try:
#                         response = requests.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'})
#                         response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
#                         result = response.json()
#                         response_text = result['candidates'][0]['content']['parts'][0]['text']
#                         break # Success, exit the loop
#                     except requests.exceptions.RequestException as e:
#                         retries += 1
#                         wait_time = 2 ** retries # Exponential backoff
#                         tqdm.write(f"API call failed: {e}. Retrying in {wait_time} seconds...")
#                         time.sleep(wait_time)
#                     except Exception as e:
#                         tqdm.write(f"An unexpected error occurred: {e}")
#                         response_text = "// Error: Failed to generate driver code."
#                         break # Non-recoverable error, break loop
                
#                 item['driver_code'] = response_text
#                 new_data.append(item)

#             # Checkpoint the data
#             if len(new_data) >= checkpoint_interval:
#                 for d in new_data:
#                     f.write(json.dumps(d) + '\n')
#                 new_data = []

#         # Write any remaining data points
#         for d in new_data:
#             f.write(json.dumps(d) + '\n')
    
#     print("Driver code generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ParlayLib code by translating from OpenMP using the GenerativeAI SDK.")

    # Use subparsers to handle different modes (merge, shuffle, split)
    subparsers = parser.add_subparsers(dest='command', required=True, help='sub-command help')

    translate_parser = subparsers.add_parser('translate', help='Translate the omp_standalone.jsonl to parlaylib code.')
    translate_parser.add_argument('--model', type=str, default="gemini-2.5-flash", help="The model to use.")
    translate_parser.add_argument('--input', type=str, default='../omp_data/omp_forpair.jsonl', help="Input dataset file (OpenMP code)")
    translate_parser.add_argument('--output', type=str, default='parlay_forpair.jsonl', help="Output file for generated data")
    translate_parser.add_argument("--api-key", type=str, required=True, help="Google AI API key.")
    translate_parser.add_argument('--temperature', type=float, default=0.2, help="Sampling temperature")
    translate_parser.add_argument('--top_p', type=float, default=0.95, help="Top-p sampling")
    translate_parser.add_argument('--top_k', type=int, default=40, help="Top-k sampling")
    translate_parser.add_argument('--n_samples', type=int, default=3, help="Number of candidates to generate per prompt.")
    
    process_parser = subparsers.add_parser('process', help='Process translated parlay code from the JSONL file')
    process_parser.add_argument('--input', type=str, default='parlay_forpair_nocomments.jsonl', help="Input dataset file (OpenMP code)")
    process_parser.add_argument('--output', type=str, default='parlay_forpair_filtered.jsonl', help="Output file for filtered data")

    process_parser = subparsers.add_parser('create_driver', help='Create driver code for the parlay code from the JSONL file')
    process_parser.add_argument('--input', type=str, default='parlay_forpair_filtered.jsonl', help="Input dataset file")
    process_parser.add_argument('--output', type=str, default='parlay_forpair_filtered_driver.jsonl', help="Output file for filtered data with driver code")
    process_parser.add_argument('--model', choices=["gemini-2.5-flash", "gemini-2.5-flash-lite"], required=True, help="Model")
    process_parser.add_argument('--api', required=True, help="The API key to use with gemini.")

    args = parser.parse_args()

    if args.command == 'translate':
        translate_main(args)
    elif args.command == 'process':
        warnings.filterwarnings("ignore", category=FutureWarning, message="`encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `RobertaSdpaSelfAttention.forward`.")
        process_translated_data_points(args.input, args.output)
    elif args.command == 'create_driver':
        with open("driver_code_template.C", 'r') as template_file:
            template = template_file.read()
        create_driver_code_data_points(args.input, args.output, template, args.model, args.api)
        


