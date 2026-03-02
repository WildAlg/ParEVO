import json
import argparse
import random
import os
from pathlib import Path
import re

def merge_jsonl_files(input_files, output_file):
    """
    Merges multiple JSONL files into a single JSONL file.

    Args:
        input_files (list): A list of paths to the input JSONL files.
        output_file (str): The path to the output JSONL file.
    """
    print(f"Merging {len(input_files)} files into {output_file}...")
    with open(output_file, 'w') as outfile:
        for file_path in input_files:
            try:
                with open(file_path, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
            except FileNotFoundError:
                print(f"Warning: File not found - {file_path}. Skipping.")
            except Exception as e:
                print(f"An error occurred while processing {file_path}: {e}")
    print(f"Successfully merged files into {output_file}")


def shuffle_jsonl_file(input_file, output_file):
    """
    Shuffles the lines of a single JSONL file and writes them to a new file.

    Args:
        input_file (str): The path to the input JSONL file.
        output_file (str): The path to the output JSONL file.
    """
    print(f"Shuffling lines from {input_file} and saving to {output_file}...")
    try:
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
        
        # Shuffle the list of lines in place
        random.shuffle(lines)
        
        with open(output_file, 'w') as outfile:
            outfile.writelines(lines)
        
        print(f"Successfully shuffled and saved to {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file not found - {input_file}.")
    except Exception as e:
        print(f"An error occurred while shuffling {input_file}: {e}")


def split_jsonl_file(input_file, train_ratio, eval_ratio, test_ratio, seed):
    """
    Splits a single JSONL file into train, eval, and test sets.

    Args:
        input_file (str): The path to the input JSONL file.
        train_ratio (float): The ratio for the training set (e.g., 0.8).
        eval_ratio (float): The ratio for the evaluation set (e.g., 0.1).
        test_ratio (float): The ratio for the test set (e.g., 0.1).
    """
    if not (train_ratio + eval_ratio + test_ratio == 1.0):
        print("Error: The sum of the ratios must be 1.0.")
        return

    print(f"Splitting {input_file} into train, eval, and test sets...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    # with open(input_file, 'r') as f:
    #     lines = f.readlines()
    
    random.seed(seed)
    random.shuffle(lines)
    total_lines = len(lines)
    
    train_end = int(total_lines * train_ratio)
    eval_end = train_end + int(total_lines * eval_ratio)
    
    train_data = lines[:train_end]
    eval_data = lines[train_end:eval_end]
    test_data = lines[eval_end:]

    # Prepare output paths
    base_path = Path(input_file)
    stem = base_path.stem  # without .jsonl
    train_file = base_path.parent / f"{stem}_train.jsonl"
    eval_file = base_path.parent / f"{stem}_eval.jsonl"
    test_file = base_path.parent / f"{stem}_test.jsonl"

    # Create the directory if it doesn't exist

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    save_jsonl(train_data, train_file)
    save_jsonl(eval_data, eval_file)
    save_jsonl(test_data, test_file)

    print(f"Saved {len(train_data)} records to {train_file}")
    print(f"Saved {len(eval_data)} records to {eval_file}")
    print(f"Saved {len(test_data)} records to {test_file}")
    

def remove_test_assertions(code):
    """
    Removes test assertions and the 'Test passed!' message from a C++ code string.
    This function specifically targets lines containing `assert()` and `std::cout << "Test passed!"`.
    """
    lines = code.split('\n')
    new_lines = []
    in_assertion_block = False
    
    prev_strip = None
    for line in lines:
        stripped_line = line.strip()

        if prev_strip is not None and 'assert(' in prev_strip and '}' in stripped_line:
            continue
        
        # Check for the start and end of the LLM-generated code block
        if '// begin of LLM generated code' in stripped_line:
            in_assertion_block = True
            new_lines.append(line)
        elif '// end of LLM generated code' in stripped_line:
            in_assertion_block = False
            new_lines.append(line)
        elif not in_assertion_block:
            # If not in the generated code block, check for assertion-related lines
            if 'assert(' in stripped_line:
                if 'for' in prev_strip:
                    new_lines.pop()
                continue
            if 'Test passed' in stripped_line:
                continue
            # if 'assert(' in stripped_line or 'Test passed' in stripped_line:
            #     continue
            new_lines.append(line)
        else:
            new_lines.append(line)

        prev_strip = stripped_line
            
    return '\n'.join(new_lines)


def remove_comments(code):
    """
    Removes C++ style single-line (//) and multi-line (/* ... */) comments from a string.
    """
    # Regex to match multi-line comments /* ... */
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Regex to match single-line comments //
    code = re.sub(r'//.*', '', code)
    return code

def process_remove_comments(input_file, field: str, output_file):
    """
    Reads a .jsonl file, removes all comments and writes the modified objects to a new .jsonl file.
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Parse the JSON object from the line
                data = json.loads(line)

                # Check if the given field exists
                if field in data:                    
                    # Step 1: Remove all comments
                    data[field] = remove_comments(data[field])
                    
                    # Step 2: Remove any empty lines that might have been created
                    data[field] = '\n'.join(
                        [l for l in data[field].split('\n') if l.strip()]
                    )

                # Write the modified JSON object to the output file
                outfile.write(json.dumps(data) + '\n')
        
        print(f"File '{input_file}' has been processed. The modified data is saved in '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'. Please check the file format.")


def process_jsonl_file(input_file, output_file):
    """
    Reads a .jsonl file, first removes test assertions, then removes all comments from
    the 'test_code' field, and writes the modified objects to a new .jsonl file.
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Parse the JSON object from the line
                data = json.loads(line)

                # Check if the 'test_code' field exists
                if 'test_code' in data:
                    # Step 1: Remove test assertions and success messages
                    data['test_code'] = remove_test_assertions(data['test_code'])
                    
                    # Step 2: Remove all comments
                    data['test_code'] = remove_comments(data['test_code'])
                    
                    # Step 3: Remove any empty lines that might have been created
                    data['test_code'] = '\n'.join(
                        [l for l in data['test_code'].split('\n') if l.strip()]
                    )

                # Write the modified JSON object to the output file
                outfile.write(json.dumps(data) + '\n')
        
        print(f"File '{input_file}' has been processed. The modified data is saved in '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'. Please check the file format.")

def add_hollow_code_attribute(input_file: str, output_file: str):
    """
    Reads a JSONL file, creates a new attribute 'test_code_hollow' for each
    JSON object, which contains the original 'test_code' with the generated
    code block removed.

    The code block to be removed is between the markers:
    "// begin of LLM generated code" and "// end of LLM generated code"
    """
    # This regex pattern matches the specific comments and the content between them,
    # using capturing groups to keep the markers. The 'flags=re.DOTALL' makes the '.'
    # character match newline characters as well.
    pattern = re.compile(r"(// begin of LLM generated code\n)(.*?)(\n // end of LLM generated code)", re.DOTALL)
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Process each line (which is a single JSON object) in the input file
        for line in infile:
            try:
                data = json.loads(line)
                
                # Check if 'test_code' exists in the JSON object
                if 'test_code' in data and isinstance(data['test_code'], str):
                    # Use re.sub to find the pattern and replace the content
                    # between the markers with nothing, effectively hollowing it out.
                    # The markers are kept using backreferences \1 and \3.
                    modified_test_code = re.sub(pattern, r"\1\3", data['test_code'])
                    data['test_code_hollow'] = modified_test_code
                
                # Write the modified JSON object to the new file
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
            
            except json.JSONDecodeError as e:
                print(f"Skipping a malformed JSON line: {line.strip()} - Error: {e}")
                continue

def process_parlaylib_header_files(source_dir: str, output_file: str):
    """
    Iterates through all .h files in a directory, extracts specific comments
    as 'instruction', the rest of the code as 'output' (with all comments
    removed), and the filename as 'name'.

    The data is then saved to a JSONL file.
    
    Args:
        source_dir (str): The path to the directory containing the .h files.
        output_file (str): The path for the output JSONL file.
    """
    # Regex pattern to find the instruction block.
    # It looks for the start and end markers and captures the content in between.
    instruction_pattern = re.compile(
        r"// \*+"  # Matches "// *****" at the start
        r"[\r\n]+"  # Matches one or more newlines
        r"(.*?)"   # Captures the instruction content (non-greedy)
        r"[\r\n]+"  # Matches one or more newlines
        r"// \*+",  # Matches "// *****" at the end
        re.DOTALL | re.MULTILINE
    )

    # Regex pattern to find all single-line and multi-line comments.
    # This will be used to remove all comments for the 'output' attribute.
    comment_pattern = re.compile(
        r"//.*?$|/\*.*?\*/",  # Matches single-line comments or multi-line comments
        re.DOTALL | re.MULTILINE
    )

    source_path = Path(source_dir)
    if not source_path.is_dir():
        print(f"Error: Directory not found at {source_dir}")
        return

    # Create a new JSONL file to write the output.
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Use os.walk to find all files in the directory and its subdirectories
        for root, _, files in os.walk(source_dir):
            for filename in files:
                if filename.endswith('.h'):
                    file_path = os.path.join(root, filename)
                    
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        
                        # Find the instruction using the instruction_pattern
                        instruction_match = instruction_pattern.search(content)
                        instruction = instruction_match.group(1).strip() if instruction_match else ""
                        
                        # Remove all comments for the 'output' content
                        output_content = re.sub(comment_pattern, "", content)
                        
                        # Create the dictionary for this file
                        data = {
                            "instruction": instruction,
                            "output": output_content,
                            "name": filename
                        }
                        
                        # Write the JSON object to the output file, followed by a newline
                        json.dump(data, outfile)
                        outfile.write('\n')
    
    print(f"Successfully processed files and saved data to {output_file}")


if __name__ == "__main__":
    # Create the main parser
    parser = argparse.ArgumentParser(description="A utility for working with JSONL files.")
    
    # Use subparsers to handle different modes (merge, shuffle, split)
    subparsers = parser.add_subparsers(dest='command', required=True, help='sub-command help')

    # Create the parser for the "merge" command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple JSONL files')
    merge_parser.add_argument("input_files", nargs='+', help="A list of JSONL files to merge.")
    merge_parser.add_argument("-o", "--output", required=True, help="The path to the output merged JSONL file.")

    # Create the parser for the "shuffle" command
    shuffle_parser = subparsers.add_parser('shuffle', help='Shuffle a single JSONL file')
    shuffle_parser.add_argument("input_file", help="The path to the JSONL file to shuffle.")
    shuffle_parser.add_argument("-o", "--output", required=True, help="The path to the output shuffled JSONL file.")
    
    # Create the parser for the "split" command
    split_parser = subparsers.add_parser('split', help='Split a single JSONL file into train, eval, and test sets')
    split_parser.add_argument("input_file", help="Path to the JSONL file")
    split_parser.add_argument("--train_ratio", type=float, default=0.96, help="Proportion of data for training")
    split_parser.add_argument("--eval_ratio", type=float, default=0.02, help="Proportion of data for evaluation")
    split_parser.add_argument("--test_ratio", type=float, default=0.02, help="Proportion of data for testing")
    split_parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")

    remove_assertion = subparsers.add_parser('rm_assert', help='Process a .jsonl file to remove test assertions and comments from the `test_code` field.')
    remove_assertion.add_argument("input_file", type=str, help="The path to the input .jsonl file.")
    remove_assertion.add_argument("-o", "--output_file", type=str, help="The path to the output .jsonl file.")

    remove_llm = subparsers.add_parser('rm_llm', help='Process a .jsonl file to remove LLM generated code from the `test_code` field.')
    remove_llm.add_argument("input_file", type=str, help="The path to the input .jsonl file.")
    remove_llm.add_argument("-o", "--output_file", type=str, help="The path to the output .jsonl file.")

    rm_comments_parser = subparsers.add_parser('rm_comments', help='Remove comments.')
    rm_comments_parser.add_argument("input_file", type=str, help="The path to the input .jsonl file.")
    rm_comments_parser.add_argument("--field", type=str, help="The field of the jsonl file that needs to be removed comments.")
    rm_comments_parser.add_argument("-o", "--output_file", type=str, help="The path to the output .jsonl file.")

    parlay_header = subparsers.add_parser('parlay_header', help='Process a .jsonl file to remove LLM generated code from the `test_code` field.')
    parlay_header.add_argument("input_dir", type=str, help="The path to the input .jsonl file.")
    parlay_header.add_argument("-o", "--output_file", type=str, help="The path to the output .jsonl file.")


    args = parser.parse_args()

    if args.command == 'merge':
        merge_jsonl_files(args.input_files, args.output)
    elif args.command == 'shuffle':
        shuffle_jsonl_file(args.input_file, args.output)
    elif args.command == 'split':
        split_jsonl_file(args.input_file, args.train_ratio, args.eval_ratio, args.test_ratio, args.seed)
    elif args.command == 'rm_assert':
        process_jsonl_file(args.input_file, args.output_file)
    elif args.command == 'rm_llm':
        add_hollow_code_attribute(args.input_file, args.output_file)
    elif args.command == 'rm_comments':
        process_remove_comments(args.input_file, args.field, args.output_file)
    elif args.command == 'parlay_header':
        process_parlaylib_header_files(args.input_dir, args.output_file)

