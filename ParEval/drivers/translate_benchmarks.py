import os
import time
import json
import argparse
import google.generativeai as genai
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# PASTE YOUR API KEY HERE OR SET IT AS AN ENV VARIABLE
API_KEY = os.environ.get("GOOGLE_API_KEY") 

# Paths relative to where you run the script
SRC_DIR = Path("cpp/benchmarks")
DEST_DIR = Path("rust/benchmarks")
PROMPTS_FILE = Path("../prompts/rust-prompts.json")

# Model (Fixed name to a valid existing model)
MODEL_NAME = "gemini-3-pro-preview" 

# ==============================================================================
# PROMPT TEMPLATE
# ==============================================================================

# We define the raw template here. It will be formatted in setup_api with examples.
SYSTEM_INSTRUCTION_TEMPLATE = """
You are an expert in High-Performance Computing, C++, and Rust.
Your task is to translate a C++ benchmark driver (cpu.cc) and its sequential logic (baseline.hpp) into a single, standalone Rust `main.rs` file.

The Rust file must:
1. Use `rayon` for parallelism, e.g. `use rayon::prelude::*;`.
2. Use the `ParEvalBenchmark` trait structure from `crate::pareval_runner`.
3. Translate the C++ `validate` logic faithfully to Rust.
4. Translate the sequential baseline algorithm from C++ to a Rust function.
5. Place `// LLM_OUTPUT_HERE` inside the parallel solution function.

### Example Translation

**C++ Input (cpu.cc)**:
{example_cpp_cpu}

**C++ Input (baseline.hpp)**:
{example_cpp_baseline}

**Rust Output (main.rs)**:
{example_rust_main}

The output must be ONLY the Rust code. Do not include markdown formatting like ```rust.
"""

USER_PROMPT_TEMPLATE = """
Here is the C++ code for the benchmark "{benchmark_name}".

### Target Rust Signature
You MUST use this exact signature for the solution function:
{rust_prompt}

### C++ Source
--- cpu.cc ---
{cpu_cc}

--- baseline.hpp ---
{baseline_hpp}

### Instructions
Generate the corresponding `main.rs`.
1. **Imports**: `use crate::pareval_runner::{{ParEvalBenchmark, run}};`, `use rayon::prelude::*;`, `use rand::prelude::*;`.
2. **Constants**: `DRIVER_PROBLEM_SIZE` is read from env vars in the runner, but usually defined as a default `const` in `main.rs` for standalone testing. Define `const DRIVER_PROBLEM_SIZE: usize = ...;` based on the C++ code.
3. **The Solution**: Use the exact signature provided above.
4. **The Baseline**: Translate logic from `baseline.hpp`.
5. **Validation**: Translate `validate` logic faithfully.
6. **Main**: `fn main() {{ run::<Context>(); }}`.

Output raw Rust code only.
"""

# ==============================================================================
# SCRIPT LOGIC
# ==============================================================================

def read_file(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def load_prompts_map(json_path):
    """
    Returns a dict mapping 'problem_type' or 'name' -> 'prompt' string
    """
    if not json_path.exists():
        print(f"[WARNING] Prompts file not found at {json_path}")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON at {json_path}")
        return {}
        
    prompt_map = {}
    for entry in data:
        # We try to match by name (e.g., "00_dense_la_lu_decomp")
        # If that's missing, you might fallback to problem_type
        p_name = entry.get("name")
        prompt_str = entry.get("prompt", "")
        if p_name:
            prompt_map[p_name] = prompt_str
    return prompt_map

def setup_api():
    if not API_KEY:
        print("Error: GOOGLE_API_KEY not found.")
        print("Please set it in the script or export it: export GOOGLE_API_KEY='your_key'")
        exit(1)
    
    genai.configure(api_key=API_KEY)

    # Load examples for Few-Shot Prompting
    # Adjust these paths to point to a valid EXISTING benchmark to use as an example
    ex_rust_path = DEST_DIR / "dense_la" / "00_dense_la_lu_decomp"/ "main.rs"
    print("ex_rust_path: ", ex_rust_path)
    ex_cpu_path = SRC_DIR / "dense_la" / "00_dense_la_lu_decomp" / "cpu.cc"
    ex_base_path = SRC_DIR / "dense_la" / "00_dense_la_lu_decomp" / "baseline.hpp"

    example_rust_main = read_file(ex_rust_path)
    example_cpp_cpu = read_file(ex_cpu_path)
    example_cpp_baseline = read_file(ex_base_path)

    if not example_rust_main:
        print("[INFO] Example files not found. Proceeding with generic instructions only.")
        formatted_instruction = SYSTEM_INSTRUCTION_TEMPLATE.replace("{example_rust_main}", "// Example not loaded")
    else:
        formatted_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(
            example_rust_main=example_rust_main,
            example_cpp_cpu=example_cpp_cpu,
            example_cpp_baseline=example_cpp_baseline
        )

    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=formatted_instruction
    )

def clean_output(text):
    text = text.strip()
    if text.startswith("```rust"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def process_benchmark(model, category, bench_name, prompt_map):
    print(f"Processing: {category}/{bench_name}...")
    
    # 1. Read Source Files
    bench_dir = SRC_DIR / category / bench_name
    cpu_cc_path = bench_dir / "cpu.cc"
    baseline_hpp_path = bench_dir / "baseline.hpp"

    if not cpu_cc_path.exists():
        print(f"  [SKIP] No cpu.cc found for {bench_name}")
        return

    cpu_cc_content = read_file(cpu_cc_path)
    baseline_hpp_content = read_file(baseline_hpp_path)

    # 2. Get Signature from JSON
    rust_prompt = prompt_map.get(bench_name, "// Signature not found in JSON")

    # 3. Generate Prompt
    prompt = USER_PROMPT_TEMPLATE.format(
        benchmark_name=bench_name,
        rust_prompt=rust_prompt,
        cpu_cc=cpu_cc_content,
        baseline_hpp=baseline_hpp_content
    )

    # 4. Call Gemini
    try:
        response = model.generate_content(prompt)
        rust_code = clean_output(response.text)
    except Exception as e:
        print(f"  [ERROR] Failed to generate code for {bench_name}: {e}")
        return

    # 5. Save to Destination
    target_bench_dir = DEST_DIR / category / bench_name
    target_bench_dir.mkdir(parents=True, exist_ok=True)
    
    target_file = target_bench_dir / "main.rs"
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(rust_code)
    
    print(f"  [SUCCESS] Written to {target_file}")

def main():
    if not SRC_DIR.exists():
        print(f"Error: Source directory {SRC_DIR} does not exist.")
        exit(1)

    model = setup_api()
    prompt_map = load_prompts_map(PROMPTS_FILE)

    # 1. List Categories (dense_la, graph, etc.)
    categories = [d.name for d in SRC_DIR.iterdir() if d.is_dir()]
    categories.sort()

    print(f"Found {len(categories)} categories in {SRC_DIR}")

    # 2. Iterate Categories and Benchmarks
    for category in categories:
        cat_path = SRC_DIR / category
        benchmarks = [d.name for d in cat_path.iterdir() if d.is_dir()]
        benchmarks.sort()
        
        for bench_name in benchmarks:
            if bench_name == "00_dense_la_lu_decomp" or bench_name == "01_dense_la_solve":
                # Skip examples used in few-shot
                continue
            process_benchmark(model, category, bench_name, prompt_map)
            # Sleep slightly to avoid hitting rate limits
            time.sleep(2) 

if __name__ == "__main__":
    main()