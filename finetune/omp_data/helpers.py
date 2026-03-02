import json
import argparse
import os
import re
from datasets import load_dataset

# Regex to find OpenMP pragmas
PRAGMA_PATTERN = re.compile(r'#pragma\s+.*?(?=\\n)', re.IGNORECASE)


def load_cpp_dataset():
    """
    Loads the HPC_Fortran_CPP dataset from Hugging Face and returns the
    'cpp' column.
    
    Returns:
        list: A list of strings containing the C++ code from the dataset.
    """
    print("Loading dataset from Hugging Face...")
    try:
        ds = load_dataset("HPC-Forran2Cpp/HPC_Fortran_CPP")
        # Extract the 'cpp' column from the 'train' split.
        cpp_data = ds['train']['cpp']
        print("Dataset loaded successfully.")
        return cpp_data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def process_cpp_dataset(cpp_data, output_prefix):
    """
    Processes a list of C++ code strings, extracts pragmas, and saves
    parallel and non-parallel entries to separate JSONL files.

    Args:
        cpp_data (list): A list of strings containing C++ code.
        output_prefix (str): The prefix for the output filenames.
    """
    non_parallel_count = 0
    parallel_count = 0
    total_count = 0

    parallel_output_file = f"{output_prefix}_parallel.jsonl"
    non_parallel_output_file = f"{output_prefix}_non_parallel.jsonl"
    
    print(f"Saving parallel pragmas to: {parallel_output_file}")
    print(f"Saving non-parallel pragmas to: {non_parallel_output_file}")

    try:
        with open(parallel_output_file, 'a') as parallel_outfile:            
            for idx, code_block in enumerate(cpp_data):
                total_count += 1
                pragmas = PRAGMA_PATTERN.findall(code_block)
                
                if pragmas:
                    # Iterate through all pragmas found in the code block
                    all_pragmas = [prg.strip() for prg in pragmas if prg.strip()]
                    new_data = {
                                "pragma": all_pragmas,
                                "code": code_block
                            }
                    json.dump(new_data, parallel_outfile)
                    parallel_outfile.write('\n')
                    parallel_count += 1
                else:
                    non_parallel_count += 1
    
    except Exception as e:
        print(f"An unexpected error occurred while writing to the output files: {e}")
    
    print("\n--- Summary ---")
    print(f"Total C++ entries processed: {total_count}")
    print(f"Parallel pragmas saved: {parallel_count}")
    print(f"Non-parallel pragmas saved: {non_parallel_count}")

def process_jsonl_file(input_files, output_prefix):
    """
    Reads a list of JSONL files, processes each line, and saves the 
    parallel and non-parallel pragma entries into two separate JSONL files.

    Args:
        input_files (list): A list of paths to the input JSONL files.
        output_prefix (str): The prefix for the output filenames. The two 
                             files will be named <prefix>_parallel.jsonl 
                             and <prefix>_non_parallel.jsonl.
    """
    non_parallel_count = 0
    parallel_count = 0
    total_count = 0
    
    parallel_output_file = f"{output_prefix}_parallel.jsonl"
    non_parallel_output_file = f"{output_prefix}_non_parallel.jsonl"
    
    print(f"Saving parallel pragmas to: {parallel_output_file}")
    print(f"Saving non-parallel pragmas to: {non_parallel_output_file}")

    try:
        with open(parallel_output_file, 'a') as parallel_outfile, \
             open(non_parallel_output_file, 'a') as non_parallel_outfile:
            
            for input_file in input_files:
                try:
                    with open(input_file, 'r') as infile:
                        for line in infile:
                            data = json.loads(line)
                            
                            if "pragma" in data and "code" in data:
                                total_count += 1
                                if data["pragma"] is not None and "parallel" in data["pragma"]:
                                    new_data = {
                                        "pragma": [data["pragma"]],
                                        "code": "#pragma omp " + data["pragma"].strip() + "\n" + data["code"]
                                    }
                                    json.dump(new_data, parallel_outfile)
                                    parallel_outfile.write('\n')
                                    parallel_count += 1
                                else:
                                    new_data = {
                                        "pragma": [data["pragma"]],
                                        "code": data["code"]
                                    }
                                    json.dump(new_data, non_parallel_outfile)
                                    non_parallel_outfile.write('\n')
                                    non_parallel_count += 1
                            
                except FileNotFoundError:
                    print(f"Error: The file {input_file} was not found and was skipped.")
                except json.JSONDecodeError:
                    print(f"Error: Failed to decode JSON from {input_file}. Check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred while writing to the output files: {e}")
    
    print("\n--- Summary ---")
    print(f"Total entries processed: {total_count}")
    print(f"Parallel pragmas saved: {parallel_count}")
    print(f"Non-parallel pragmas saved: {non_parallel_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL files or Hugging Face dataset to split into parallel and non-parallel pragma files.")
    parser.add_argument("-o", "--output_prefix", help="The prefix for the new output JSONL files.")
    parser.add_argument("-i", "--input_files", nargs='+', required=True, help="The paths to the input JSONL files.")
    parser.add_argument("--dataset", action="store_true", help="Use the HPC-Forran2Cpp dataset instead of local files.")
    
    args = parser.parse_args()
    
    if args.dataset:
        process_jsonl_file(args.input_files, args.output_prefix)
        cpp_data = load_cpp_dataset()
        if cpp_data:
            process_cpp_dataset(cpp_data, args.output_prefix)
    else:
        process_jsonl_file(args.input_files, args.output_prefix)