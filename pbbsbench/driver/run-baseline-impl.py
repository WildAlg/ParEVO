import os
import subprocess
import argparse
import json
import sys

def run_single_benchmark(base_dir, alg_dir, server, threads, rounds):
    """
    Automates the process of building and running baseline benchmarks for
    a single algorithm directory across multiple thread counts.

    Args:
        base_dir (str): The base directory containing all the algorithm folders.
        alg_dir (str): The specific algorithm directory to process.
        server (str): The name of the server where the tests are run.
        threads (list): A list of integer thread counts to test.
    """
    alg_path = os.path.join(base_dir, alg_dir)
    alg_name = alg_dir.split('/', 1)[1]
    baseline_runtime_path = os.path.join(alg_path, "baseline_runtime")

    # Check if the algorithm directory exists before proceeding
    if not os.path.isdir(alg_path):
        print(f"Error: Algorithm directory '{alg_path}' not found. Skipping.")
        return

    # New feature: Check if baseline_runtime directory already exists
    if os.path.isdir(baseline_runtime_path) and args.rerun:
        print(f"Skipping '{alg_dir}' as the baseline_runtime directory already exists.")
        return

    print(f"Processing algorithm directory: {alg_dir}")

    # Create the baseline_runtime directory if it doesn't exist
    os.makedirs(baseline_runtime_path, exist_ok=True)

    # Clean previous builds
    try:
        print("  -> Cleaning previous builds...")
        subprocess.run("make cleanall", shell=True, check=True, cwd=alg_path, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  -> Warning: 'make cleanall' failed. Stderr: {e.stderr}")
    
    # Run tests for each thread count
    for thread_count in threads:
        print(f"  -> Running tests with {thread_count} threads.")

        # ParlayLib tests
        parlay_log_filename_large = f"{alg_name}_parlay_{server}_large_{thread_count}.log"
        parlay_log_file_path_large = os.path.join(baseline_runtime_path, parlay_log_filename_large)
        parlay_log_filename_small = f"{alg_name}_parlay_{server}_small_{thread_count}.log"
        parlay_log_file_path_small = os.path.join(baseline_runtime_path, parlay_log_filename_small)


        try:
            print("    -> Building ParlayLib version...")
            subprocess.run("make", shell=True, check=True, cwd=alg_path, capture_output=True, text=True)
            
            print("    -> Running large input test...")
            test_command_large = f"./testInputs -r {rounds} -p {thread_count}"
            test_result_large = subprocess.run(test_command_large, shell=True, check=True, cwd=alg_path, capture_output=True, text=True)
            full_output_large = f"Build Output:\n\nTest Output:\n{test_result_large.stdout}\n{test_result_large.stderr}\n"
            with open(parlay_log_file_path_large, "w") as f:
                f.write(full_output_large)
            print(f"      -> Successfully saved large test output to {parlay_log_file_path_large}")

            print("    -> Running small input test...")
            test_command_small = f"./testInputs_small -r {rounds} -p {thread_count}"
            test_result_small = subprocess.run(test_command_small, shell=True, check=True, cwd=alg_path, capture_output=True, text=True)
            full_output_small = f"Build Output:\n\nTest Output:\n{test_result_small.stdout}\n{test_result_small.stderr}\n"
            with open(parlay_log_file_path_small, "w") as f:
                f.write(full_output_small)
            print(f"      -> Successfully saved small test output to {parlay_log_file_path_small}")

        except subprocess.CalledProcessError as e:
            print(f"      -> Error: Command failed with exit code {e.returncode}. Output saved to log.")
            with open(parlay_log_file_path_large, "a") as f: # Use 'a' for append to not overwrite
                f.write(f"Error executing command: {e.cmd}\n")
                f.write(f"Stdout:\n{e.stdout}\n")
                f.write(f"Stderr:\n{e.stderr}\n")
        except FileNotFoundError:
            print(f"      -> Error: Executable not found. Make sure it was built correctly.")
        except Exception as e:
            print(f"      -> An unexpected error occurred: {e}")

        # Clean after Parlay tests
        try:
            subprocess.run("make cleanall", shell=True, check=True, cwd=alg_path, capture_output=True, text=True)
        except:
            pass

        # OpenMP tests
        omp_log_filename_large = f"{alg_name}_omp_{server}_large_{thread_count}.log"
        omp_log_file_path_large = os.path.join(baseline_runtime_path, omp_log_filename_large)
        omp_log_filename_small = f"{alg_name}_omp_{server}_small_{thread_count}.log"
        omp_log_file_path_small = os.path.join(baseline_runtime_path, omp_log_filename_small)


        try:
            print("    -> Building OpenMP version...")
            subprocess.run("make OPENMP=1", shell=True, check=True, cwd=alg_path, capture_output=True, text=True)
            
            print("    -> Running large input test...")
            if args.server in ["XXXXXX", "xxxxx"]:
                test_command_large = f"srun --exclusive --mem=64G -N 1 -n 1 -c {thread_count} ./testInputs -r {rounds} -p {thread_count}"
            else:
                test_command_large = f"./testInputs -r {rounds} -p {thread_count}"
            print("cmd: ", test_command_large)
            test_result_large = subprocess.run(test_command_large, shell=True, check=True, cwd=alg_path, capture_output=True, text=True)
            full_output_large = f"Build Output:\n\nTest Output:\n{test_result_large.stdout}\n{test_result_large.stderr}\n"
            with open(omp_log_file_path_large, "w") as f:
                f.write(full_output_large)
            print(f"      -> Successfully saved large test output to {omp_log_file_path_large}")

            print("    -> Running small input test...")
            if args.server in ["XXXXXX", "xxxxx"]:
                test_command_small = f"srun --exclusive --mem=12G -N 1 -n 1 -c {thread_count} ./testInputs_small -r {rounds} -p {thread_count}"
            else:
                test_command_small = f"./testInputs_small -r {rounds} -p {thread_count}"
            test_result_small = subprocess.run(test_command_small, shell=True, check=True, cwd=alg_path, capture_output=True, text=True)
            full_output_small = f"Build Output:\n\nTest Output:\n{test_result_small.stdout}\n{test_result_small.stderr}\n"
            with open(omp_log_file_path_small, "w") as f:
                f.write(full_output_small)
            print(f"      -> Successfully saved small test output to {omp_log_file_path_small}")
        
        except subprocess.CalledProcessError as e:
            print(f"      -> Error: Command failed with exit code {e.returncode}. Output saved to log.")
            with open(omp_log_file_path_large, "a") as f: # Use 'a' for append
                f.write(f"Error executing command: {e.cmd}\n")
                f.write(f"Stdout:\n{e.stdout}\n")
                f.write(f"Stderr:\n{e.stderr}\n")
        except FileNotFoundError:
            print(f"      -> Error: Executable not found. Make sure it was built correctly.")
        except Exception as e:
            print(f"      -> An unexpected error occurred: {e}")

        # Clean after OpenMP tests
        try:
            subprocess.run("make cleanall", shell=True, check=True, cwd=alg_path, capture_output=True, text=True)
        except:
            pass

    print("-" * 50)

def main(args):
    """
    Main function to parse arguments and start the benchmark process.
    """

    # Load JSON data
    try:
        with open(args.input_json, "r") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The JSON file '{args.input_json}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{args.input_json}' is not a valid JSON file.")
        sys.exit(1)

    print(f"Starting benchmark runs from JSON file: {args.input_json}")
    print("-" * 50)

    # Iterate through each entry in the JSON data
    if isinstance(json_data, list):
        for entry in json_data:
            try:
                alg_dir = entry["name"]
                run_single_benchmark(args.base_dir, alg_dir, args.server, args.threads, args.rounds)
            except KeyError:
                print("Skipping an entry in JSON due to missing 'name' key.")
                continue
    else:
        # Handle the case where the JSON is a single object, not a list
        try:
            alg_dir = json_data["name"]
            run_single_benchmark(args.base_dir, alg_dir, args.server, args.threads, args.rounds)
        except KeyError:
            print("Error: JSON object is missing the 'name' key.")
    
    print("All benchmark runs completed.")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate baseline benchmark runs from a JSON input file.")
    parser.add_argument("--input-json", "-i", type=str, required=True, help="Path to the JSON file containing the list of algorithms.")
    parser.add_argument("--base-dir", "-b", type=str, default="../benchmarks", help="The base directory where algorithm folders are located.")
    parser.add_argument("--server", required=True, type=str, help="The server name to run testInputs.")
    parser.add_argument("-t", "--threads", nargs="+", default=[1,2,4,8,16,32,36,48,64], type=int, help="Number of threads to run testInputs.")
    parser.add_argument("-r", "--rounds", default=5, type=int, help="Number of rounds to run testInputs.")
    parser.add_argument("--rerun", action="store_true", help="Rerun all benchmarks even if baseline_runtime directory exists.")
    
    args = parser.parse_args()

    main(args)