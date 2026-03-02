""" Create a set of tests from the serial benchmarks in the drivers.
    author: Daniel Nichols
    date: November 2023
"""
# std imports
from argparse import ArgumentParser
import glob
import json
from os import PathLike
from os.path import join as path_join, exists as path_exists
import re


def get_file_contents(fpath: PathLike) -> str:
    with open(fpath, 'r') as f:
        return f.read()

def get_substr_after_first_of(s: str, substr: str) -> str:
    """ Return the substring in s after the first instance of substr. """
    return s[s.find(substr) + len(substr):]

def get_return_type(code: str, use_parlay: bool = False) -> str:
    """ First identify the line that has a function definition, then return the return type. """
    lines = code.split('\n')
    in_parlay_block = False
    
    for line in lines:
        # Track if we're in a Parlay-specific block
        if '#ifdef USE_PARLAY' in line:
            in_parlay_block = True
        elif '#endif' in line:
            in_parlay_block = False
        
        # If we want Parlay implementation, only consider lines in Parlay block
        # If we want standard implementation, skip lines in Parlay block
        if use_parlay and not in_parlay_block:
            continue
        if not use_parlay and in_parlay_block:
            continue
            
        if "NO_INLINE correct" in line and line.strip().endswith(') {'):
            return line.split()[0]

def extract_implementation(baseline: str, use_parlay: bool = False) -> str:
    """ Extract the implementation of the correct function.
        If use_parlay is True, extract the Parlay version.
        Otherwise, extract the standard C++ version.
    """
    lines = baseline.split('\n')
    in_parlay_block = False
    found_function = False
    implementation_lines = []
    brace_count = 0
    
    for i, line in enumerate(lines):
        # Track Parlay blocks
        if '#ifdef USE_PARLAY' in line:
            in_parlay_block = True
            continue
        elif '#endif' in line:
            in_parlay_block = False
            continue
        
        # Check if we should consider this line based on use_parlay setting
        if use_parlay and not in_parlay_block:
            continue
        if not use_parlay and in_parlay_block:
            continue
        
        # Look for function definition
        if not found_function and "NO_INLINE correct" in line and '(' in line:
            found_function = True
            # Find the opening brace
            if '{' in line:
                brace_count = 1
                # Extract everything after the opening brace
                impl_start = line.find('{') + 1
                impl_text = line[impl_start:]
                if impl_text.strip():
                    implementation_lines.append(impl_text)
            continue
        
        if found_function:
            implementation_lines.append(line)
            brace_count += line.count('{')
            brace_count -= line.count('}')
            
            # When we've closed all braces, we're done
            if brace_count == 0:
                # Remove the last closing brace
                last_line = implementation_lines[-1]
                close_brace_idx = last_line.rfind('}')
                if close_brace_idx != -1:
                    implementation_lines[-1] = last_line[:close_brace_idx]
                break
    
    return '\n'.join(implementation_lines)

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('benchmarks_root', help='Root directory of the benchmarks')
    parser.add_argument('prompts', help='Path to prompts json')
    parser.add_argument('output', help='Json output path')
    parser.add_argument('--use-parlay', action='store_true', 
                        help='Extract Parlay implementation instead of standard C++ implementation')
    args = parser.parse_args()

    with open(args.prompts, 'r') as f:
        prompts = json.load(f)
    
    output = []
    for prompt in prompts:
        baseline_fpath = path_join(args.benchmarks_root, prompt['problem_type'], prompt['name'], 'baseline.hpp')

        if prompt['parallelism_model'] != 'serial' or not path_exists(baseline_fpath):
            continue

        baseline = get_file_contents(baseline_fpath)
        impl = extract_implementation(baseline, use_parlay=args.use_parlay)
        return_type = get_return_type(baseline, use_parlay=args.use_parlay)
        
        prompt['outputs'] = [
            impl, 
            ' }' if return_type == 'void' else ' return 0; }', 
            ' undefinedFunction(); }'
        ]
        output.append(prompt)

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    main()