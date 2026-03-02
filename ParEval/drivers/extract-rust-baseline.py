"""
Extract baseline Rust functions from benchmark source files
Use the extracted code to test the correctness of generated rust main.rs for each problem.
"""
import os
import re
import json
import argparse
from pathlib import Path

# Configuration - Adjust these paths if necessary
DEFAULT_PROMPTS_FILE = "../prompts/rust-prompts.json"
DEFAULT_BENCHMARK_DIR = "rust/benchmarks"
DEFAULT_OUTPUT_FILE = "rust-prompts-with-serial-sol.json"

def extract_inner_body(content, start_index):
    """
    Finds the first '{' after start_index and extracts content until the matching '}'.
    Removes the trailing newline before the closing brace to match formatting.
    """
    open_brace_index = content.find('{', start_index)
    if open_brace_index == -1:
        return None

    count = 1
    current_index = open_brace_index + 1
    
    while count > 0 and current_index < len(content):
        char = content[current_index]
        if char == '{':
            count += 1
        elif char == '}':
            count -= 1
        current_index += 1

    if count == 0:
        # Extract content strictly between { and }
        body = content[open_brace_index + 1 : current_index - 1]
        
        # Remove the specific trailing newline if present
        if body.endswith('\n'):
            body = body[:-1]
            
        return body
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Update rust-prompts.json with extracted baseline code.")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_FILE, help="Path to input rust-prompts.json")
    parser.add_argument("--bench_dir", default=DEFAULT_BENCHMARK_DIR, help="Root directory of Rust benchmarks")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Path to save the new JSON file")
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    bench_dir = Path(args.bench_dir)
    output_path = Path(args.output)

    if not prompts_path.exists():
        print(f"Error: Prompts file not found at {prompts_path}")
        return

    # 1. Load the Input JSON
    print(f"Loading prompts from {prompts_path}...")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    # Create a lookup map for faster access: name -> dictionary entry
    prompt_map = {entry["name"]: entry for entry in prompts_data if "name" in entry}

    # 2. Walk the benchmark directory
    print(f"Scanning benchmarks in {bench_dir}...")
    fn_pattern = re.compile(r"fn\s+(correct_\w+)\s*<*.*?>*\s*\(")
    
    updated_count = 0

    for dirpath, _, filenames in os.walk(bench_dir):
        if "main.rs" in filenames:
            bench_name = os.path.basename(dirpath)
            
            # Check if this benchmark is in our JSON file
            if bench_name in prompt_map:
                file_path = os.path.join(dirpath, "main.rs")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Find the baseline function
                    match = fn_pattern.search(content)
                    if match:
                        func_name = match.group(1)
                        start_pos = match.start()
                        
                        # Extract body
                        body = extract_inner_body(content, start_pos)
                        
                        if body:
                            # Update the JSON object in memory
                            prompt_map[bench_name]["outputs"] = [body]
                            updated_count += 1
                            print(f"  [+] Extracted '{bench_name}' (found {func_name})")
                        else:
                            print(f"  [-] Failed to parse body for '{bench_name}'")
                    else:
                        print(f"  [-] No correct_* function found for '{bench_name}'")
                except Exception as e:
                    print(f"  [!] Error reading {file_path}: {e}")

    # 3. Save to a NEW JSON file
    print(f"\nSaving updated data to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts_data, f, indent=4)

    print(f"Done. Updated {updated_count} entries.")

if __name__ == "__main__":
    main()

# import os
# import re
# import json
# import argparse

# def extract_inner_body(content, start_index):
#     """
#     Starting from the function definition at start_index, this finds the 
#     first opening brace '{' and extracts everything strictly inside 
#     the matching closing brace '}'.
#     """
#     # 1. Find the first opening brace after the function name
#     open_brace_index = content.find('{', start_index)
#     if open_brace_index == -1:
#         return None

#     # 2. Walk forward counting braces to find the matching close
#     count = 1
#     current_index = open_brace_index + 1
    
#     while count > 0 and current_index < len(content):
#         char = content[current_index]
#         if char == '{':
#             count += 1
#         elif char == '}':
#             count -= 1
#         current_index += 1

#     if count == 0:
#         # current_index is now just after the closing '}'
#         # We want the content *between* open_brace_index and (current_index - 1)
        
#         # open_brace_index + 1 skips the '{'
#         # current_index - 1 is the closing '}' index
#         body = content[open_brace_index + 1 : current_index - 1]
        
#         # 3. Clean up: The user requested "remove the last \n}"
#         # Usually the body ends with "\n   ", so we strip the trailing newline
#         # to leave the last inner brace as the final character.
#         if body.endswith('\n'):
#             body = body[:-1]
            
#         return body
    
#     return None

# def scan_benchmarks(root_dir):
#     results = {}
    
#     # Regex to find 'fn correct_something' (the signature start)
#     fn_pattern = re.compile(r"fn\s+(correct_\w+)\s*<*.*?>*\s*\(")

#     for dirpath, _, filenames in os.walk(root_dir):
#         if "main.rs" in filenames:
#             file_path = os.path.join(dirpath, "main.rs")
#             benchmark_name = os.path.basename(dirpath)
            
#             try:
#                 with open(file_path, "r", encoding="utf-8") as f:
#                     content = f.read()

#                 # Find all 'correct_' functions
#                 match = fn_pattern.search(content)
#                 if match:
#                     func_name = match.group(1)
#                     start_pos = match.start()
                    
#                     # Extract only the inner body
#                     inner_code = extract_inner_body(content, start_pos)
                    
#                     if inner_code:
#                         results[benchmark_name] = {
#                             "function_name": func_name,
#                             "code": inner_code
#                         }
#                         print(f"[+] Extracted body for {benchmark_name} ({func_name})")
#                     else:
#                         print(f"[-] Error: Could not parse body for {benchmark_name}")
#                 else:
#                     # Optional: notify if no baseline is found
#                     # print(f"[-] No baseline found in {benchmark_name}")
#                     pass

#             except Exception as e:
#                 print(f"[!] File error {file_path}: {e}")

#     return results

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("root_dir", help="Path to the benchmarks folder")
#     parser.add_argument("--output", default="baseline_functions.json", help="Output JSON file")
#     args = parser.parse_args()

#     data = scan_benchmarks(args.root_dir)

#     with open(args.output, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=4)
    
#     print(f"\nSaved {len(data)} benchmarks to {args.output}")