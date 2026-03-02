# This script generates a supervised fine-tuning dataset from ParlayLib tests
# using the Google GenerativeAI SDK instead of raw REST API calls.

import sys
sys.path.append("..")
import re
import json
from pathlib import Path
from utils import generate_response
import time
import argparse
import google.generativeai as genai


def extract_json_from_response(text):
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1)
    return json.loads(text)


def load_dataset(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def ask_gemini_to_decompose(data_point, args):
    return None

def ask_gemini_for_similar(data_point, args):
    # we use this prompt to generate the original 14k similar parlaylib datapoints
    # prompt = (
    #     "You are an expert in parallel computing and in the ParlayLib library."
    #     "Given the following data point:\n"
    #     f"{json.dumps(data_point, indent=2)}\n\n"
    #     f"Please generate {args.num_generated} more data points that are similar in style, structure, and content."
    #     " Return them as a JSON array."
    # )

    # 
    prompt = (
        "You are an expert in parallel computing and in the ParlayLib library."
        "Given the following data point:\n"
        f"{json.dumps(data_point, indent=2)}\n\n"
        f"Please generate {args.num_generated} more data points that use similar primitives and have educational value for learning the ParlayLib library."
        f"You can be as creative as you wish so long as you write good, clean, compilable, and correct code."
        f"Include necessary header files, such as `<parlay/primitives.h>` and `<parlay/parallel.h>`."
        f"Make sure the code you return follows the format below:"
        """{{
            "instruction": "<Instruction>",
            "input": "<Input>",
            "output": "<.cpp code (give me the code but not wrapped in ```cpp ```), but without any test assertions, static_asserts. The code should be clean and ready to run.>",
            "primitive": "<Primitive>",
            "test_name": "<Test Name>",
            "test_code": "#include <Respective Header File>\n#include <iostream>\n#include <parlay/primitives.h>\n\nint main() {{\n // begin of LLM generated code\n\n // end of LLM generated code\n\n // Add test assertions here, using `assert()` and not `ASSERT_EQ`\n std::cout << "Test passed!" << std::endl;}}"
        }}"""
        " Return them as a JSON array."
    )

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "candidate_count": args.n_samples,
    }

    model = genai.GenerativeModel(args.model)
    response = model.generate_content(prompt, generation_config=generation_config)
    return response


def main(args):
    genai.configure(api_key=args.api_key)
    dataset = load_dataset(args.input)
    all_generated = []

    for idx, data_point in enumerate(dataset):
        if (idx <= 40): continue
        print(f"Processing data point {idx+1}/{len(dataset)}...")
        try:
            response = ask_gemini_for_similar(data_point, args)

            # Add a one-second delay after each API call
            time.sleep(12)

            for j, candidate in enumerate(response.candidates):
                generated_text = candidate.content.parts[0].text
                try:
                    generated_data = extract_json_from_response(generated_text)
                    all_generated.append({
                        "original": data_point,
                        "generated": generated_data
                    })
                    with open(f'parsed_{args.output}', 'a') as f:
                        for obj in generated_data:
                            f.write(json.dumps(obj) + '\n')
                except Exception:
                    print("Warning: Could not parse Gemini response as JSON. Saving raw text.")
                    all_generated.append({
                        "original": data_point,
                        "generated": generated_text
                    })

            time.sleep(1)  # avoid overwhelming the API
        except Exception as e:
            print(f"Error processing data point {idx}: {e}")
            continue

    with open(args.output, 'a') as f:
        for item in all_generated:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or parse Gemini examples using the GenerativeAI SDK.")
    parser.add_argument('--model', choices=["gemini-2.5-flash", "gemini-2.5-pro"], required=True, help="The model to use.")
    parser.add_argument('--input', type=str, default='parlaylib_test_dataset_all.jsonl', help="Input dataset file")
    parser.add_argument('--output', type=str, default='gemini_generated_similar_data.jsonl', help="Output file for generated data")
    parser.add_argument("--api-key", type=str, required=True, help="Google AI API key.")
    parser.add_argument('--num_generated', type=int, default=10, help="Number of similar data points to generate for each input")
    parser.add_argument('--temperature', type=float, default=0.2, help="Sampling temperature")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p sampling")
    parser.add_argument('--top_k', type=int, default=40, help="Top-k sampling")
    parser.add_argument('--n_samples', type=int, default=5, help="Number of candidates to generate")
    args = parser.parse_args()

    main(args)
