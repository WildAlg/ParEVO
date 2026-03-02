# This script generates a supervised fine-tuning dataset from ParlayLib tests.
# The generated dataset is stored in `parlaylib_test_dataset.jsonl`.
import sys
sys.path.append("..")
import re
import json
from pathlib import Path
from utils import generate_response
import time

API_KEY = "XXXX"
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17" 
PROMPT = """
Given this ParlayLib test:
- test suite: {}
- test name {}
- test body ```cpp{}```, turn it into a high-quality instruction-style supervised fine-tuning dataset entry, \ 
tailored for teaching an LLM how to apply ParlayLib's primitive in algorithmic problems.
Follow the format below:
{{
    "instruction": "<Instruction>",
    "input": "<Input>",
    "output": "<.cpp code adapted from the test body (give me the code but not wrapped in ```cpp ```), but \ 
    without any test assertions, static_asserts, or includes. The code should be clean and ready to run.>",
    "primitive": "<Primitive>",
    "test_name": "<Test Name>"
}}
""" 

def extract_tests(content):
    # Match each TEST(...) { ... } block including multi-line bodies
    pattern = re.compile(r'TEST\(([^,]+),\s*([^)]+)\)\s*\{(.*?)^\}', re.DOTALL | re.MULTILINE)
    return pattern.findall(content)

def clean_code(code):
    """Remove test assertions, static_asserts, includes, and indentation."""
    code = re.sub(r'#include\s+[<"].*[>"]', '', code)
    code = re.sub(r'ASSERT_.*?;', '', code)
    code = re.sub(r'static_assert.*?;', '', code)
    code = re.sub(r'^\s*//.*$', '', code, flags=re.MULTILINE)
    return code.strip()

def generate_instruction(code_block, primitive):
    # A simple prompt for now; you can customize per primitive
    return f"Use ParlayLib's `{primitive}` primitive to solve the following problem.\n\n{code_block}"

def generate_output(code_block):
    # Clean the code block for "expected output" (i.e., solution)
    return clean_code(code_block)

def extract_json_from_response(response):
    json_str = re.sub(r'^```json\n|\n```$', '', response.strip())
    return json_str

def get_valid_response(prompt, model, api_key, max_retries=3):
    for attempt in range(1, max_retries + 1):
        response = generate_response(prompt, model, api_key)
        if response is None:
            return 1
        # print(f"Attempt {attempt}: {response}")
        try:
            json_str = extract_json_from_response(response)
            python_object = json.loads(json_str)
            return python_object
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print("Retrying...")
                time.sleep(1)
            else:
                print("Max retries reached. Exiting.")
                return None



def parse_test_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    tests = extract_tests(content)
    dataset = []

    count = 0
    for suite_name, test_name, test_body in tests:
        # if count % 6 == 0:
        #     time.sleep(40) # Rate limit to avoid hitting the API too fast
        print(f"Count: {count}, Processing test suite: {suite_name}, test name: {test_name}")
        # print(f"Test body:\n{test_body}\n")
        local_prompt = PROMPT.format(suite_name, test_name, test_body)
        # print(f"prompt:\n{local_prompt}\n")

        response_object = get_valid_response(local_prompt, GEMINI_MODEL, API_KEY)
        if response_object == 1:
            print("Failed to get a valid response after retries.")
            return dataset

        # print(f"Response:\n{response_object}\n")


        dataset.append(response_object)
        count += 1

    return dataset

# append to jsonl file
def save_to_jsonl(data, output_path):
    with open(output_path, 'a') as f:
        for entry in data:
            f.write(json.dumps(entry, indent=None) + '\n')

# === Example usage ===
if __name__ == "__main__":
    test_dir = Path("../../parlaylib/test/")
    # test_file = "../../parlaylib/test/test_monoid.cpp"  # Replace with the actual path to your test file
    output_path = "parlaylib_test_dataset.jsonl"

    all_data = []
    processed_files = {}
    start_time = time.time()
    # processed_files = {"test_monoid.cpp", "test_padded.cpp", "test_delayed_sequence.cpp", "test_separate_compilation1.cpp", "test_delayed_map.cpp", "test_file_map.cpp", "test_parsing.cpp", "test_sequence.cpp", "test_hash_table.cpp", "test_delayed_filter_op.cpp", "test_quicksort.cpp", "test_collect_reduce.cpp", "test_quicksort.cpp", "test_sorting_primitives.cpp", "test_delayed_reduce.cpp"}
    for test_file in test_dir.glob("*.cpp"):
        if test_file.name in processed_files:
            continue
        print(f"Processing file: {test_file}")
        dataset = parse_test_file(test_file)
        # all_data.extend(dataset)
        save_to_jsonl(dataset, output_path)
    

    # dataset = parse_test_file(test_file)
    # save_to_jsonl(all_data, output_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to process all files: {elapsed_time:.2f} seconds")
    # print(f"Extracted {len(all_data)} examples to {output_path}")
