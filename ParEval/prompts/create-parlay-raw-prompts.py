import os
import re
import google.generativeai as genai

# --- User-provided LLM utility functions ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
API_KEY = "YOUR_API_KEY" # !!! IMPORTANT: REPLACE WITH YOUR ACTUAL API KEY !!!
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

def generate_response(prompt, model_name, api_key):
    """
    Generates a response from the Gemini model.
    Make sure your API_KEY is correctly set and has access to the Gemini API.
    """
    if api_key == "YOUR_GEMINI_API_KEY":
        raise ValueError("API_KEY not set. Please replace 'YOUR_GEMINI_API_KEY' with your actual API key.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    try:
        # Use a higher temperature for more creative/diverse translations, or lower for more deterministic.
        # Max output tokens to avoid excessively long outputs.
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.2, # Lower temperature for code translation accuracy
            max_output_tokens=2048 # Adjust as needed for typical function sizes
        ))
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None
# --- End of User-provided LLM utility functions ---

def extract_code(text, filename):
    # Regular expression to find C++ code blocks
    code_blocks = re.findall(r'#start(.*?)#end', text, re.DOTALL)
    
    # If code blocks are found, return the first one (assuming there's only one)
    if code_blocks:
        cpp_code = code_blocks[0].strip()
        with open(filename, 'w') as file:
            file.write(cpp_code)
    else:
        return "No C++ code found."

def transform_omp_to_parlay(input_file_path, output_file_path):
    """
    Reads an OpenMP C++ file, replaces "OpenMP" with "parlaylib",
    and uses an LLM to specifically transform all function definitions
    to use parlaylib primitives. Saves the result to a new 'parlay' file.
    """
    print(f"Processing: {input_file_path}")
    
    with open(input_file_path, 'r') as f:
        original_content = f.read()

    # print("original content: ", original_content)
    # # Initial string replacement across the entire file
    # modified_content = original_content.replace("OpenMP", "parlaylib")

    # --- Few-shot example for the LLM ---
    # This example guides the LLM on the desired transformation style.
    # It's crucial for good results, even for general functions.
    few_shot_openmp_example = """
/* Count the number of doubles in the vector x that have a fractional part 
   in [0, 0.25), [0.25, 0.5), [0.5, 0.75), and [0.75, 1). Store the counts in `bins`.
   Use OpenMP to compute in parallel.
*/
void exampleCountQuartiles(std::vector<double> const& x, std::array<size_t, 4> &bins) {
"""
    few_shot_parlay_example = """
/* Count the number of doubles in the vector x that have a fractional part 
   in [0, 0.25), [0.25, 0.5), [0.5, 0.75), and [0.75, 1). Store the counts in `bins`.
   Use ParlayLib to compute in parallel.
*/
void exampleCountQuartiles(parlay::sequence<double> const& x, std::array<size_t, 4> &bins) {
"""

    # functions = find_all_function_defs(original_content)
    
    # Store replacements to apply them after finding all functions
    # This prevents issues with spans changing due to replacements
    # replacements = [] 

    # if not functions:
    #     print("  - No function definitions found. Only performing string replacement.")

    # for func_info in functions:
    #     original_func_text = func_info['original_text']
    #     func_span = func_info['span']

        # Constructing the prompt for the current function
    prompt = f"""
You are a C++ code translator. Your task is to translate the provided C++ function definition into a function that uses ParlayLib for parallelism. If you see "OpenMP" in the comment above the function definition, you should replace it with "ParlayLib".

**Instructions:**
1.  **Crucially, change `std::vector` to `parlay::sequence` in function parameters and variable declarations where it makes sense for parallel operations.**
2.  **Do NOT include any `#include` directives in your response.** Assume necessary ParlayLib and standard headers (e.g., `parlay/parallel.h`, `parlay/primitives.h`, `<cmath>`, `<array>`) will be handled by the overall compilation unit.
3.  **Return only the translated function definition in ParlayLib and not the entire implementation.**
4.  **Wrap your answer in #start and #end**.

**Example Transformation (to guide your translation style):**

Example 1:
OpenMP Function:
```cpp
{few_shot_openmp_example}
```
ParlayLib Function:
```cpp
{few_shot_parlay_example}
```

Now, please translate the following OpenMP C++ function to ParlayLib:

OpenMP Function:
```cpp
{original_content}
```
""" 
        # print(f"  - Sending function to LLM (starts at char {func_span[0]}): \n{original_func_text[:100]}...\n")
        
    translated_func_text = generate_response(prompt, DEFAULT_GEMINI_MODEL, API_KEY)

    if translated_func_text:
        print("  - LLM translation successful for function.")
        # replacements.append((func_span[0], func_span[1], translated_func_text))
    else:
        print("  - LLM translation failed for function. Keeping original.")
    extract_code(translated_func_text, output_file_path)
    print(f"  - Saved transformed file to: {output_file_path}")

def transform_omp_to_rust(input_file_path, output_file_path):
    """
    Reads an OpenMP C++ file, replaces "OpenMP" with "Rust Rayon",
    and uses an LLM to specifically transform all function definitions
    to use rust primitives. Saves the result to a new 'rust' file.
    """
    print(f"Processing: {input_file_path}")
    
    with open(input_file_path, 'r') as f:
        original_content = f.read()

    # print("original content: ", original_content)
    # # Initial string replacement across the entire file
    # modified_content = original_content.replace("OpenMP", "parlaylib")

    # --- Few-shot example for the LLM ---
    # This example guides the LLM on the desired transformation style.
    # It's crucial for good results, even for general functions.
    few_shot_openmp_example = """
/* Count the number of doubles in the vector x that have a fractional part 
   in [0, 0.25), [0.25, 0.5), [0.5, 0.75), and [0.75, 1). Store the counts in `bins`.
   Use OpenMP to compute in parallel.
*/
void exampleCountQuartiles(std::vector<double> const& x, std::array<size_t, 4> &bins) {
"""
    few_shot_rust_example = """
/* Count the number of doubles in the vector x that have a fractional part 
   in [0, 0.25), [0.25, 0.5), [0.5, 0.75), and [0.75, 1). Store the counts in `bins`.
   Use Rust Rayon to compute in parallel.
*/
pub fn exampleCountQuartiles(x: &[f64], bins: &mut [usize; 4]) {
"""

    prompt = f"""
You are a C++ code translator. Your task is to translate the provided C++ function definition into a function that uses Rust Rayon for parallelism. If you see "OpenMP" in the comment above the function definition, you should replace it with "Rust Rayon".

**Instructions:**
1.  **Do NOT include any `#include` directives in your response.** Assume necessary standard headers will be handled by the overall compilation unit.
2.  **Return only the translated function definition in Rust and not the entire implementation.**
3.  **Wrap your answer in #start and #end**.

**Example Transformation (to guide your translation style):**

Example 1:
OpenMP Function:
```cpp
{few_shot_openmp_example}
```
Rust Function:
```cpp
{few_shot_rust_example}

Now, please translate the following OpenMP C++ function to Rust:

OpenMP Function:
```cpp
{original_content}
```
""" 
        # print(f"  - Sending function to LLM (starts at char {func_span[0]}): \n{original_func_text[:100]}...\n")
        
    translated_func_text = generate_response(prompt, DEFAULT_GEMINI_MODEL, API_KEY)

    if translated_func_text:
        print("  - LLM translation successful for function.")
        # replacements.append((func_span[0], func_span[1], translated_func_text))
    else:
        print("  - LLM translation failed for function. Keeping original.")
    extract_code(translated_func_text, output_file_path)
    print(f"  - Saved transformed file to: {output_file_path}")

def process_raw_directory(root_dir="raw", language="parlay"):
    """
    Iterates through all subdirectories of 'root_dir' and processes 'omp' files.
    """
    if not os.path.isdir(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "omp":
                input_file_path = os.path.join(dirpath, filename)
                if language == "parlay":
                    output_file_path = os.path.join(dirpath, "parlay")
                    transform_omp_to_parlay(input_file_path, output_file_path)
                elif language == "rust":
                    output_file_path = os.path.join(dirpath, "rust")
                    transform_omp_to_rust(input_file_path, output_file_path)

if __name__ == "__main__":
    # --- IMPORTANT ---
    # 1. Replace "YOUR_GEMINI_API_KEY" with your actual Google Gemini API key.
    #    You can get one from Google AI Studio.
    # 2. Ensure your API key has access to the Gemini API (e.g., gemini-1.5-flash).
    # 3. Ensure your 'raw' directory structure is set up correctly.
    #    The dummy setup below helps for initial testing.
    # --- ------------- ---

    # Create a dummy 'raw' directory structure and an 'omp' file for testing
#     dummy_dir = "raw_test/histogram/24_histogram_count_quartile_test"
#     os.makedirs(dummy_dir, exist_ok=True)
#     dummy_omp_content = """
# /* Count the number of doubles in the vector x that have a fractional part 
#    in [0, 0.25), [0.25, 0.5), [0.5, 0.75), and [0.75, 1). Store the counts in `bins`.
#    Use OpenMP to compute in parallel.
#    Examples:
#    input: [7.8, 4.2, 9.1, 7.6, 0.27, 1.5, 3.8]
#    output: [2, 1, 2, 2]
#    input: [1.9, 0.2, 0.6, 10.1, 7.4]
#    output: [2, 1, 1, 1]
# */
# #include <vector>
# #include <array>
# #include <cmath> // For floor
# #include <omp.h> // Example OpenMP include

# void countQuartiles(std::vector<double> const& x, std::array<size_t, 4> &bins) {
    
# """
    # with open(os.path.join(dummy_dir, "omp"), "w") as f:
    #     f.write(dummy_omp_content)
    # print("Dummy 'raw' directory and 'omp' file created for demonstration.")
    # print("-" * 30)

    # Run the processing
    language = "rust"
    process_raw_directory("raw", "rust")

    print("-" * 30)
    print("Transformation complete. Check the 'raw' directory for 'parlay' files.")
    print("\nIMPORTANT: LLMs can sometimes produce incorrect or suboptimal code. ALWAYS review the generated 'parlay' files for correctness, performance, and adherence to ParlayLib best practices.")