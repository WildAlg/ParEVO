# This script parses a C++ file to extract function definitions
# and saves them into a JSONL (JSON Lines) file.
# Each line in the output file represents a single function.

import re
import json
import argparse
from pathlib import Path

# A simplified regex pattern to capture function definitions.
# This pattern is not exhaustive and might miss some edge cases
# like functions with templates or complex macros, but it works for
# many common function styles.
# It looks for:
# 1. The entire function definition (`full_function`)
# 2. A return type (`return_type`)
# 3. A function name (`name`)
# 4. A parameter list in parentheses (`parameters`)
# 5. A function body in curly braces (`body`)
# The `re.DOTALL` flag allows `.` to match newlines, which is crucial for
# matching the function body.
FUNCTION_PATTERN = re.compile(
    r'(?P<full_function>\s*(?P<return_type>[\w\s\*&<>:]+)\s+(?P<name>[\w:]+)\s*\((?P<parameters>[^)]*)\)\s*\{(?P<body>[\s\S]*?)\n\s*\})',
    re.DOTALL
)

def parse_cpp_functions(file_content):
    """
    Parses a string containing C++ code and extracts all function definitions.
    
    Args:
        file_content (str): The C++ code as a single string.
        
    Returns:
        list: A list of dictionaries, where each dictionary represents a function.
    """
    functions = []
    
    # Find all matches of the function pattern in the file content.
    # `re.finditer` is used to get match objects with named groups.
    for match in FUNCTION_PATTERN.finditer(file_content):
        # Extract the named groups from the regex match.
        functions.append({
            "function": match.group("full_function").strip(),
            "return_type": match.group("return_type").strip(),
            "name": match.group("name").strip(),
            "parameters": match.group("parameters").strip(),
            "body": match.group("body").strip()
        })
        
    return functions


def save_to_jsonl(data, output_file):
    """
    Saves a list of dictionaries to a JSONL file.
    
    Args:
        data (list): A list of dictionaries to save.
        output_file (str): The path to the output JSONL file.
    """
    with open(output_file, 'w') as f:
        for item in data:
            # `json.dumps` serializes the dictionary to a JSON string.
            f.write(json.dumps(item) + '\n')


def main():
    """Main function to parse the C++ file and save the output."""
    parser = argparse.ArgumentParser(description="Parse C++ functions and save to a JSONL file.")
    parser.add_argument('--input', type=str, required=True, help="Input C++ file path.")
    parser.add_argument('--output', type=str, default='output.jsonl', help="Output JSONL file path.")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.")
        return

    with open(input_path, 'r') as f:
        cpp_code = f.read()

    print(f"Parsing functions from {args.input}...")
    functions = parse_cpp_functions(cpp_code)

    if functions:
        save_to_jsonl(functions, args.output)
        print(f"Successfully extracted {len(functions)} functions and saved to {args.output}.")
    else:
        print("No functions were found in the provided C++ file.")


if __name__ == "__main__":
    main()
