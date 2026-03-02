import os
import json
import re
import argparse

INDEX_OFFSET_MAP = {
    "claude": 20,
    "gpt": 40,
    "gemini": 60,
    "deepseek": 80,
    "gemini-2.5-pro-finetuned": 200,
    "gemini-3-pro-preview-0.2": 300,
    "gemini-3-pro-preview-0.7": 400
}

def main(args):
    # Load JSON data
    with open(args.input, "r") as file:
        json_data = json.load(file)

    for idx, data in enumerate(json_data):
        print(f"Processing {idx+1} data.")

        alg_name = data["alg_name"]
        alg_dir = data["name"]
        try: 
            outputs = data["outputs"]
        except:
            print("No `outputs` found for this json object, skipping.")
            continue
        generate_model = data["generate_model"]
        if generate_model not in INDEX_OFFSET_MAP.keys():
            generate_model = generate_model.split("-")[0]
        offset = INDEX_OFFSET_MAP[generate_model]

        # Define the directory for this algorithm
        destination_dir = os.path.join(args.output_dir, alg_dir)
        os.makedirs(destination_dir, exist_ok=True)

        for jdx, output in enumerate(outputs):
            output_filename = f"{alg_name}Mix{jdx+offset}.h"  # start index=3
            file_path = os.path.join(destination_dir, output_filename)

            # Extract the code from the JSON output using regex
            code_match = (
                re.search(r'#start(.*?)#end', output, re.DOTALL) or
                re.search(r'```cpp(.*?)```', output, re.DOTALL) or
                re.search(r'```(.*?)```', output, re.DOTALL)  # catch plain triple backticks
            )

            if code_match:
                cpp_code = code_match.group(1).strip()
            else:
                print("No C++ code block found in the JSON output, using the raw output.")
                cpp_code = output

            # Double-check: strip ```cpp ... ``` wrappers if still present
            fence_match = re.match(r'^\s*```(?:cpp)?\s*(.*?)\s*```$', cpp_code, re.DOTALL)
            if fence_match:
                cpp_code = fence_match.group(1).strip()
            try:
                with open(file_path, 'w') as f:
                    f.write(cpp_code)
                print(f"Code successfully extracted and saved to: {file_path}")
            except IOError as e:
                print(f"Error writing to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract C++ code from JSON outputs and save to files.")
    parser.add_argument("--input", "-i",type=str,required=True,help="Path to the JSON file containing generated outputs.")
    parser.add_argument("--output-dir", "-o",type=str,default="../benchmarks",help="Base directory where extracted code files will be saved.")
    args = parser.parse_args()

    main(args)
