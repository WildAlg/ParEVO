#!/usr/bin/env python3
"""
Simple script to generate Python code using Gemini API for a list of prompts.

Usage:
    python generate_generators.py
"""

import os
import re
import dotenv
from pathlib import Path
from openai import OpenAI
from get_description_editorial import fetch_dmoj_data

dotenv.load_dotenv()

# Configuration
MODEL_NAME = "gemini-3-pro-preview"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = os.environ.get("GEMINI3_API_KEY")

OUTPUT_DIR = Path("./generators")


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks or return raw text."""
    # Try to find ```python ... ``` block
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Try to find ``` ... ``` block (no language specified)
    pattern = r"```\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Return raw text if no code block found
    return text.strip()


def generate(client: OpenAI, prompt: str) -> str:
    """Make a generation request and return the response text."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=18192,
    )
    return response.choices[0].message.content


def process_problem(client: OpenAI, problem_id: str, prompt: str):
    """Generate code for a single problem and save results."""
    print(f"Processing {problem_id}...", end=" ", flush=True)
    
    try:
        raw_response = generate(client, prompt)
        python_code = extract_python_code(raw_response)
        
        # Save raw response
        raw_path = OUTPUT_DIR / f"generate_{problem_id}.txt"
        with open(raw_path, "w") as f:
            f.write(raw_response)
        
        # Save extracted Python code
        code_path = OUTPUT_DIR / f"generate_{problem_id}.py"
        with open(code_path, "w") as f:
            f.write(python_code)
        
        print("done")
        
    except Exception as e:
        print(f"error: {e}")


def get_problem_prompts(problem_ids):
    prompts = dict()
    instruction = """\
    Write a python generator of test cases to this problem. In the main function, it should call the generate function with some parameter related \
    to the problem size, and write two files (one is input, the other is expected output). If this problem does not have a unique output, \
    or the problem does not have any parameter to increase its test case size, please point this out explicitly, do not need to provide a checker or methods.
    Otherwise, in the comment of the main function, provide appropriate parameter so that the generated test case should take at least 10x more time \
    for a standard solution. The input file and output file should be specified in the main function also the test case size related parameters.
    """
    for problem_id in problem_ids:
        dmoj_data = fetch_dmoj_data(problem_id = problem_id)
        prompts[problem_id] = instruction + dmoj_data
    return prompts

def main():
    if not API_KEY:
        print("Error: GEMINI3_API_KEY not set")
        return
        
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    
    prompts = get_problem_prompts([
        'ccc15s1',
        'ccc17s3',
        'cco08p4',
        'coci11c1p3',
        'coci19c1p3',
        'coci21c2p1',
        'coci23c2p2',
        'joi21op1',
    ])
    
    for problem_id, prompt in prompts.items():
        process_problem(client, problem_id, prompt)


if __name__ == "__main__":
    main()
    # print(get_problem_prompts(['coci19c1p3'])['coci19c1p3'])