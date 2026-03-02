import os
import json
import argparse
import subprocess
import re
import shutil
import uuid
import concurrent.futures
import threading
from pathlib import Path

# --- Configuration ---
INDEX_OFFSET_MAP = {
    "claude": 20,
    "gpt": 40,
    "gemini": 60,
    "deepseek": 80,
    "gemini-2.5-pro-finetuned": 200,
    "gemini-3-pro-preview-0.2": 300,
    "gemini-3-pro-preview-0.7": 400
}

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARKS_DIR = ROOT_DIR / "benchmarks"
OUTPUT_JSON_FILENAME = "benchmarks_results.json"

CATEGORIES_WITH_ONLY_H = ["integerSort", "comparisonSort", "removeDuplicates", "nearestNeighbors"]

# Global Lock for JSON writing
json_lock = threading.Lock()

# --- Helper: In-Memory Parser ---
def parse_line_to_dict(line, current_data):
    """Parses output lines for timing data."""
    line = line.strip()
    parts = line.split(':')
    if len(parts) >= 3 and "geomean =" in parts[-1]:
        try:
            dataset_name = parts[0].strip()
            result_chunk = parts[-1]
            times_str, geomean_str = result_chunk.split("geomean =")
            geomean_val = float(geomean_str.strip())
            runtimes = [float(t.strip().replace("'", "")) for t in times_str.split(',') if t.strip()]
            current_data[dataset_name] = {"runtimes": runtimes, "geomean": geomean_val}
            return
        except ValueError:
            pass

    if "geomean of mins =" in line:
        try:
            match = re.search(r"geomean of mins\s*=\s*([\d\.]+).*geomean of geomeans\s*=\s*([\d\.]+)", line)
            if match:
                current_data["summary"] = {
                    "geomean_mins": float(match.group(1)),
                    "geomean_geomeans": float(match.group(2))
                }
        except ValueError:
            pass

# --- Helper: Save JSON ---
def save_results_safely(data, filepath):
    """Thread-safe JSON save."""
    with json_lock:
        temp_path = filepath + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, filepath)

# --- Helper: Compilation ---
def run_make_clean(destination_dir):
    """Runs make clean (less destructive than cleanall)."""
    try:
        subprocess.run(["make", "clean"], cwd=destination_dir, check=True, 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pass 

def build_benchmark_utils(benchmarks_root, alg_dir_rel):
    """Builds shared utils (like sortCheck) in the sibling 'bench' directory."""
    parts = alg_dir_rel.strip("/").split("/")
    if len(parts) < 2:
        return 

    category = parts[0]
    bench_utils_dir = os.path.join(benchmarks_root, category, "bench")
    
    if os.path.exists(bench_utils_dir):
        # We run make here to ensure checkers (sortCheck, etc) exist
        try:
            subprocess.run(["make"], cwd=bench_utils_dir, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # Sibling bench dir might not behave as expected, but we try anyway
            pass

def run_make(destination_dir, mix_value, scheduler):
    """Runs 'make' and returns Success (bool)."""
    env = os.environ.copy()
    if scheduler == "omp":
        cmd = ["make", f"MIX={mix_value}", "OPENMP=1"]
    else:
        cmd = ["make", f"MIX={mix_value}"]
    
    try:
        subprocess.run(cmd, cwd=destination_dir, timeout=30, check=True, capture_output=True, text=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Compilation Failed: {destination_dir} MIX={mix_value}\n{e.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired as e:
        print(f"   ❌ Timeout: {destination_dir} MIX={mix_value}\n{e.stderr[:200]}")
        return False

def _find_executable(destination_dir, benchmarks_root, alg_dir, input_size):
    """Finds the executable path."""
    category = alg_dir.split("/", 1)[0] if "/" in alg_dir else alg_dir
    benchmark_root = os.path.join(benchmarks_root, category)

    candidates = [destination_dir, os.path.join(destination_dir, "bench"), os.path.join(benchmark_root, "bench")]
    exe_names = ["testInputs_small", "testInputs"] if input_size == "small" else ["testInputs", "testInputs_small"]

    for d in candidates:
        for exe in exe_names:
            path = os.path.join(d, exe)
            if os.path.exists(path) and os.access(path, os.X_OK):
                return d, f"./{exe}", path 
    raise FileNotFoundError(f"Could not find testInputs in {destination_dir}")

def compute_mix_offset(generate_model: str) -> int:
    platform = generate_model or ""
    if platform not in INDEX_OFFSET_MAP.keys():
        platform = platform.split("-")[0]
    return INDEX_OFFSET_MAP.get(platform, 0)

def extract_cpp_code(raw_output):
    code_match = (re.search(r'#start(.*?)#end', raw_output, re.DOTALL) or 
                  re.search(r'```cpp(.*?)```', raw_output, re.DOTALL) or 
                  re.search(r'```(.*?)```', raw_output, re.DOTALL))
    cpp_code = code_match.group(1).strip() if code_match else raw_output
    fence_match = re.match(r'^\s*```(?:cpp)?\s*(.*?)\s*```$', cpp_code, re.DOTALL)
    if fence_match: cpp_code = fence_match.group(1).strip()
    return cpp_code

# --- WORKER FUNCTION ---
def execute_benchmark_task(task):
    """Executed by ThreadPoolExecutor."""
    cmd = task["cmd"]
    cwd_dir = task["cwd"]
    env = task["env"]
    log_file = task["log_file"]
    keep_logs = task.get("keep_logs", False)

    cmd_prefix = []
    for key in ["OPENMP", "MIX"]:
        if env.get(key, None) is not None:
            cmd_prefix.append(f"{key}={env[key]}")
    cmdstr = ' '.join(cmd_prefix + cmd)
    task_id = f"{task['alg_name']}_mix{task['mix_value']}_t{task['thread']}"
    print(f"   [START] {task_id}")

    current_json_data = {}
    run_success = False   
    check_success = False 
    captured_stdout = ""
    captured_stderr = ""

    try:
        with open(log_file, "w") as f_log:
            process = subprocess.Popen(
                cmdstr, cwd=cwd_dir, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                text=True, bufsize=1
            )
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    f_log.write(line)
                    captured_stdout += line
                    parse_line_to_dict(line, current_json_data)
            
            _, stderr_out = process.communicate()
            if stderr_out:
                f_log.write("\n--- STDERR ---\n")
                f_log.write(stderr_out)
                captured_stderr = stderr_out

            rc = process.returncode
            
            has_name_error = "NameError" in captured_stderr or "NameError" in captured_stdout

            if rc == 0:
                run_success = True
                check_success = True
            else:
                if has_name_error:
                    run_success = True
                    check_success = False
                    current_json_data["error_type"] = "Correctness Check Failed (NameError)"
                else:
                    run_success = False
                    check_success = False
                    current_json_data["error_type"] = f"Runtime Error (RC={rc})"

            if not run_success:
                print(f"   [FAIL] RC={rc} {task_id}")

    except Exception as e:
        print(f"   [ERROR] {e} in {task_id}")
        current_json_data["error"] = str(e)
        run_success = False
        check_success = False
    
    if not keep_logs and os.path.exists(log_file):
        try:
            os.remove(log_file)
        except OSError:
            pass

    return {
        "alg_index": task["alg_index"],
        "output_index": task["output_index"],
        "thread": task["thread"],
        "data": current_json_data,
        "status": {
            "build_success": True, 
            "run_success": run_success,
            "check_success": check_success
        }
    }

# --- MAIN RUNNER ---
def run_benchmarks(args):
    benchmarks_root = os.path.abspath(args.benchmarks_dir)
    
    # --- LOAD INPUT ---
    with open(args.input, "r") as file:
        json_data = json.load(file)
    
    # Normalize structure
    for entry in json_data:
        if "outputs" in entry:
            new_outputs = []
            for item in entry["outputs"]:
                if isinstance(item, str):
                    new_outputs.append({"code": item, "results": {}})
                else:
                    if "results" not in item: item["results"] = {}
                    new_outputs.append(item)
            entry["outputs"] = new_outputs

    # --- RESUME LOGIC (SMART MERGE) ---
    if os.path.exists(args.output_json):
        print(f"🔹 Found existing {args.output_json}, attempting to resume...")
        try:
            with open(args.output_json, "r") as f:
                existing_data = json.load(f)
            
            # Create a lookup map from the existing file: { "algorithm_name": entry_object }
            existing_map = { item.get("name"): item for item in existing_data }
            
            merged_count = 0
            for current_entry in json_data:
                name = current_entry.get("name")
                if name in existing_map:
                    existing_entry = existing_map[name]
                    
                    # Merge outputs
                    # We assume the order of outputs (mixes) is stable. 
                    # If unsure, we should ideally have IDs for outputs, but index is standard here.
                    current_outputs = current_entry.get("outputs", [])
                    existing_outputs = existing_entry.get("outputs", [])
                    
                    for i, out in enumerate(current_outputs):
                        if i < len(existing_outputs):
                            saved_results = existing_outputs[i].get("results", {})
                            if saved_results:
                                out["results"].update(saved_results)
                                merged_count += 1
            
            print(f"   ✅ Merged results for {merged_count} outputs from checkpoint.")

        except json.JSONDecodeError:
            print("   ⚠️  Existing JSON corrupted. Starting fresh.")
        except Exception as e:
            print(f"   ⚠️  Resume failed ({e}). Starting fresh.")

    # Apply Filter AFTER merge (so we don't lose loaded data for other algs if we save later)
    # Actually, if we filter here, we only run specific ones, but we should probably 
    # load the full file to preserve data if we are overwriting it. 
    # For now, we only process the filtered list.
    
    processing_data = json_data
    if args.benchmark:
        processing_data = [item for item in json_data if item.get("name") == args.benchmark]

    safe_bin_dir = os.path.abspath(f"temp_bin_{uuid.uuid4().hex}")
    os.makedirs(safe_bin_dir, exist_ok=True)
    print(f"🔹 Temp Binary Directory: {safe_bin_dir}")

    injected_files = []

    # --- PHASE 0: INJECT CODE ---
    print("\n🔹 Phase 0: Injecting Code")
    for data in processing_data:
        alg_name = data.get("alg_name")
        alg_dir_rel = data.get("name")
        if alg_dir_rel.split('/')[1] == "ips4o":
            print("\n! The current pipeline cannot work with ips4o, skipping it for now...")
            continue
        destination_dir = os.path.join(benchmarks_root, alg_dir_rel)
        if not os.path.exists(destination_dir): os.makedirs(destination_dir, exist_ok=True)
        
        offset = compute_mix_offset(data.get("generate_model"))
        for jdx, output_obj in enumerate(data.get("outputs", [])):
            cpp_code = extract_cpp_code(output_obj["code"])
            alg_dir_rel_start = alg_dir_rel.split('/')[0]
            if alg_dir_rel_start not in CATEGORIES_WITH_ONLY_H:
                file_path = os.path.join(destination_dir, f"{alg_name}Mix{jdx + offset}.C")
            else:
                file_path = os.path.join(destination_dir, f"{alg_name}Mix{jdx + offset}.h")
            file_path_wo_h = os.path.join(destination_dir, f"{alg_name}Mix{jdx + offset}")

            try:
                with open(file_path, 'w') as f: 
                    f.write(cpp_code)
                    injected_files.append(file_path)
                    injected_files.append(file_path_wo_h)
            except IOError as e:
                print(f"   ❌ Error writing {file_path}: {e}")

    # --- PHASE 1: COMPILE & PREPARE ---
    print("\n🔹 Phase 1: Compilation & Preparation")
    execution_queue = []

    for alg_idx, data in enumerate(processing_data):
        alg_dir_rel = data["name"]
        alg_name = data.get("alg_name", alg_dir_rel.split("/")[-1])
        destination_dir = os.path.join(benchmarks_root, alg_dir_rel)
        offset = compute_mix_offset(data["generate_model"])
        
        # 1. Clean environment
        run_make_clean(destination_dir)
        
        # 2. Build Shared Tools (Checkers) if missing
        build_benchmark_utils(benchmarks_root, alg_dir_rel)

        for out_idx, output_obj in enumerate(data.get("outputs", [])):
            mix_value = out_idx + offset
            
            # Checkpoint Check
            needed_threads = [str(t) for t in args.threads]
            existing_results = output_obj.get("results", {})
            if all(t in existing_results for t in needed_threads):
                print(f"   ⏩ Skipping {alg_name} Mix{mix_value} (All threads done)")
                continue

            # 3. Compile Mix
            print(f"   Compiling {alg_dir_rel}/{alg_name} (MIX={mix_value})...")
            build_success = run_make(destination_dir, mix_value, args.scheduler)

            if not build_success:
                for p in args.threads:
                    output_obj["results"][str(p)] = {
                        "build_success": False, "run_success": False, 
                        "check_success": False, "error": "Compilation Failed"
                    }
                # Save immediately to full json_data (not just processing_data)
                save_results_safely(json_data, args.output_json)
                continue

            # 4. Find & Copy Binary
            try:
                cwd_dir, exe_cmd, src_exe_path = _find_executable(destination_dir, benchmarks_root, alg_dir_rel, args.input_size)
                unique_name = f"{alg_name}_mix{mix_value}_{uuid.uuid4().hex[:6]}"
                safe_exe_path = os.path.join(safe_bin_dir, unique_name)
                shutil.copy(src_exe_path, safe_exe_path)
                shutil.copymode(src_exe_path, safe_exe_path)
                
                # Delete original binary
                if os.path.exists(src_exe_path): os.remove(src_exe_path)
            except Exception as e:
                print(f"     ❌ Binary error: {e}")
                for p in args.threads:
                    output_obj["results"][str(p)] = {
                        "build_success": True, "run_success": False,
                        "check_success": False, "error": f"Binary not found: {str(e)}"
                    }
                save_results_safely(json_data, args.output_json)
                continue

            # 5. Queue Tasks
            eval_output_dir = os.path.join(destination_dir, "eval-output")
            os.makedirs(eval_output_dir, exist_ok=True)

            for p in args.threads:
                if str(p) in existing_results:
                    continue

                base_filename = f"{data['generate_model']}_{mix_value}_{alg_name}_{args.scheduler}_{args.server}_{args.input_size}_{p}"
                
                exe_call = safe_exe_path 
                if args.server in ["XXXXXX", "xxxxx"]:
                    cmd = ["srun", "--export=ALL", "--exclusive", "--mem=12G" if args.input_size=="small" else "--mem=64G", 
                           "-N", "1", "-n", "1", "-c", str(p),
                           exe_call, "-r", "5", "-p", str(p)]
                else:
                    cmd = [exe_call, "-r", "5", "-p", str(p)]

                if args.input_size == "small" and "testInputs" in exe_cmd and "small" not in exe_cmd:
                    cmd.append("-s")
                if args.keep_data is True:
                    cmd.append("-k")

                env = os.environ.copy()
                env["MIX"] = str(mix_value)
                if args.scheduler == "omp": env["OPENMP"] = "1"

                # Note: We must store indices relative to processing_data for result retrieval
                execution_queue.append({
                    "cmd": cmd, "cwd": cwd_dir, "env": env,
                    "log_file": os.path.join(eval_output_dir, base_filename + ".log"),
                    "keep_logs": args.keep_logs,
                    "alg_index": alg_idx, "output_index": out_idx, "thread": p,
                    "alg_name": alg_name, "mix_value": mix_value
                })
        
        # Cleanup
        run_make_clean(destination_dir)

    # --- PHASE 2: EXECUTION ---
    print(f"\n🔹 Phase 2: Running {len(execution_queue)} tasks...")
    if execution_queue:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel_jobs) as executor:
            futures = {executor.submit(execute_benchmark_task, task): task for task in execution_queue}
            
            completed_count = 0
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                
                # Update data structure (processing_data points to objects inside json_data)
                target_output = processing_data[res["alg_index"]]["outputs"][res["output_index"]]
                combined_result = res["data"]
                combined_result.update(res["status"])
                target_output["results"][str(res["thread"])] = combined_result

                # Periodic Save (Save the FULL json_data, not just filtered)
                save_results_safely(json_data, args.output_json)
                
                completed_count += 1
                if completed_count % 5 == 0:
                    print(f"   ... Saved progress ({completed_count}/{len(execution_queue)})")
    else:
        print("   No new tasks to run.")

    # --- PHASE 3: CLEANUP & SAVE ---
    print("\n🔹 Phase 3: Cleanup and Save")
    if os.path.exists(safe_bin_dir): shutil.rmtree(safe_bin_dir)
    
    print(f"   Removing {len(injected_files)} generated header files...")
    for fpath in injected_files:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except OSError as e:
                print(f"   ⚠️ Failed to remove {fpath}: {e}")

    save_results_safely(json_data, args.output_json)
    print(f"Final results saved to {args.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output-json", "-o", type=str, default="benchmarks_results.json", help="Save the runtime in a json file.")
    parser.add_argument("--benchmarks-dir", "-d", type=str, default=str(DEFAULT_BENCHMARKS_DIR))
    parser.add_argument("--benchmark", "-b", type=str, default=None)
    parser.add_argument("-t", "--threads", nargs="+", default=[1, 2, 4, 8, 16, 32, 36, 48, 64], type=int)
    parser.add_argument("-s", "--scheduler", choices=["parlay", "omp", "cilk"], required=True)
    parser.add_argument("--server", required=True, type=str)
    parser.add_argument("--input-size", type=str, choices=["small", "large"], default="small")
    parser.add_argument("-m", "--max-parallel-jobs", type=int, default=8)
    parser.add_argument("-k", "--keep-data", action="store_true", default=True, help="Keep intermediate benchmark data files.")
    parser.add_argument("--keep-logs", action="store_true", default=False, help="Keep python execution logs in eval-output.")
    args = parser.parse_args()
    
    if not os.path.exists(args.benchmarks_dir):
        print(f"Error: Benchmarks directory not found: {args.benchmarks_dir}")
        exit(1)
    run_benchmarks(args)
