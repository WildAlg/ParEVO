import google.generativeai as genai
import re
import subprocess
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
API_KEY = "XXXXX"

def generate_response(prompt, model_name, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

def save_response_to_file(response_text, file_path):
    with open(file_path, 'w') as file:
        file.write(response_text)

def generate_response_save(prompt, model_name, api_key, file_path):
    response_text = generate_response(prompt, model_name, api_key)
    save_response_to_file(response_text, file_path)
    return response_text

def extract_driver_code(filename):
    file_path = os.path.join(ROOT_DIR, "parlaylib", "examples", filename)

    try:
        with open(file_path, 'r') as file:
            code = file.read()
            return code
            # print(code)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Modify filename to save the extracted code written by LLM in your intended directory
def extract_code(text, filename):
    # Regular expression to find C++ code blocks
    code_blocks = re.findall(r'```cpp\n(.*?)```', text, re.DOTALL)
    
    # If code blocks are found, return the first one (assuming there's only one)
    if code_blocks:
        cpp_code = code_blocks[0].strip()
        with open(filename, 'w') as file:
            file.write(cpp_code)
    else:
        return "No C++ code found."

def compile_code(filename, executable, directory):
    # Obtain the directory path of parlaylib/include
    parlaylib_include_path = os.path.join(ROOT_DIR, "parlaylib", "include")
    # print(f"Parlaylib Include Path: {parlaylib_include_path}")

    # Compile the C++ code
    compile_command = ["g++", "-std=c++17", "-I", parlaylib_include_path, filename, "-o", executable]
    print(f"Compile Command: {compile_command}")
    compile_process = subprocess.run(compile_command, capture_output=True, text=True, cwd=directory)
    return compile_process.returncode, compile_process.stderr

# This needs to be depreciated. We should write the Run command by ourselves, so that 
# we can evaluate the LLM written code with different input size, different graphs (i.e. 
# graph generated + online graph datasets)
def extract_run_command(text, executable):
    # Adjust the regular expression to match the run command format
    run_command_match = re.search(r'\./' + re.escape(executable) + r'\s+\d+', text)
    run_command = run_command_match.group(0) if run_command_match else None
    return run_command

def run_executable(run_command, executable, directory):
    # Run the compiled executable
    run_command_list = run_command.split()
    run_process = subprocess.run(run_command_list, capture_output=True, text=True, cwd=directory)
    return f"Execution Output:\n{run_process.stdout}"

def run_parlaylib_example(filename, executable, arguments, directory):
    parlaylib_include_path = os.path.join(ROOT_DIR, "parlaylib", "include")
    file_path = os.path.join(ROOT_DIR, "parlaylib", "examples", filename)
    executable = executable + "_example"

    # Compile the C++ code
    compile_command = ["g++", "-std=c++17", "-I", parlaylib_include_path, file_path, "-O3", "-o", executable]
    print(f"Compile Command: {compile_command}")
    compile_process = subprocess.run(compile_command, capture_output=True, text=True, cwd=directory)
    if compile_process.returncode == 0:
        print("Compilation Successful.")
        run_command = f"./{executable}" + " " + arguments
        print("run_command: ", run_command)
        output = run_executable(run_command, executable, directory)
        performance_output_filename = f"{executable}_example_performance.txt"
        with open(performance_output_filename, 'w') as file:
            file.write(output)
    else:
        print(f"Compilation Failed:\n{compile_process.stderr}")

def execute_command(command, directory):
    process = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=directory)
    return process.stdout, process.stderr, process.returncode

