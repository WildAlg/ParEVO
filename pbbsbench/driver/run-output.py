# run-output.py (modified: can rerun from rerun-requests.json)
import os
import json
import argparse
import subprocess

INDEX_OFFSET_MAP = {
    "claude": 20,
    "gpt": 40,
    "gemini": 60,
    "deepseek": 80,
    "gemini-2.5-pro-finetuned": 200,
    "gemini-3-pro-preview-0.2": 300,
    "gemini-3-pro-preview-0.7": 400
}

def run_make_cleanall(destination_dir):
    """Runs 'make cleanall' in the specified directory."""
    cmd = ["make", "cleanall"]
    print(f"Running in {destination_dir}: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=destination_dir, check=True,
                                capture_output=True, text=True)
        print("Make cleanall stdout:\n", result.stdout)
        print("Make cleanall stderr:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running make cleanall in {destination_dir}")
        print(e.stdout)
        print(e.stderr)
        raise

def run_make(destination_dir, mix_value, scheduler):
    """Runs 'make' with the appropriate MIX and scheduler flags."""
    if scheduler == "omp":
        cmd = ["make", f"MIX={mix_value}", "OPENMP=1"]
    elif scheduler == "parlay":
        cmd = ["make", f"MIX={mix_value}"]
    else:
        cmd = ["make", f"MIX={mix_value}"]

    print(f"Running in {destination_dir}: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=destination_dir, check=True,
                                capture_output=True, text=True)
        print("Make stdout:\n", result.stdout)
        print("Make stderr:\n", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running make in {destination_dir} with MIX={mix_value}:")
        print(e.stdout)
        print(e.stderr)
        return False

def _find_executable(destination_dir, output_dir, alg_dir, input_size):
    """
    Find testInputs / testInputs_small in common PBBSBench layouts.

    Returns: (cwd_dir, exe_name)
      - cwd_dir: directory to use as subprocess cwd
      - exe_name: "./testInputs" or "./testInputs_small"
    """
    # alg_dir like "classify/decisionTree"
    category = alg_dir.split("/", 1)[0] if "/" in alg_dir else alg_dir
    benchmark_root = os.path.join(output_dir, category)

    # Candidate directories to look in (cwd locations)
    candidates = [
        destination_dir,
        os.path.join(destination_dir, "bench"),
        os.path.join(benchmark_root, "bench"),  # e.g. ../benchmarks/classify/bench
    ]

    # Candidate executable names, prefer size-specific if requested
    if input_size == "small":
        exe_names = ["testInputs_small", "testInputs"]
    else:
        exe_names = ["testInputs", "testInputs_small"]

    for d in candidates:
        for exe in exe_names:
            path = os.path.join(d, exe)
            if os.path.exists(path) and os.access(path, os.X_OK):
                return d, f"./{exe}"

    raise FileNotFoundError(
        "Could not find testInputs executable. Tried:\n  " +
        "\n  ".join(
            os.path.join(d, exe)
            for d in candidates
            for exe in (["testInputs_small", "testInputs"])
        )
    )


def run_testInputs(output_dir, alg_dir, destination_dir, mix_value, generate_model,
                   alg_name, scheduler, server, threads, input_size):
    """Runs testInputs for given config and logs output."""
    # Log destination:
    # - baseline reruns: keep alongside baseline logs
    # - non-baseline: eval-output
    if str(mix_value) == "baseline" or str(generate_model) == "baseline":
        eval_output_dir = os.path.join(destination_dir, "baseline_runtime")
        log_ext = ".log"
    else:
        eval_output_dir = os.path.join(destination_dir, "eval-output")
        log_ext = ".txt"
    os.makedirs(eval_output_dir, exist_ok=True)

    # Find executable + cwd
    cwd_dir, exe = _find_executable(destination_dir, output_dir, alg_dir, input_size)

    for p in threads:
        log_file = os.path.join(
            eval_output_dir,
            f"{generate_model}_{mix_value}_{alg_name}_{scheduler}_{server}_{input_size}_{p}{log_ext}"
        )
        
        # Check if valid log already exists
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    content = f.read()
                if "GLIBCXX_3.4.32" not in content and "TEST TERMINATED ABRUPTLY" not in content:
                    print(f"Skipping (valid log exists): {log_file}")
                    continue
            except Exception as e:
                print(f"Error reading log file, will rerun: {e}")
        
        print(f"Logging to: {log_file}")

        # Build cmd:
        # Construct srun command
        if args.server in ["XXXXXX", "xxxxx"]:
            cmd = []
            if args.input_size == "small":
                cmd = [
                    "srun", "--exclusive", "--mem=12G", "-N", "1", "-n", "1", "-c", str(p),
                    exe, "-r", "5", "-p", str(p)
                ]
            elif args.input_size == "large":
                cmd = [
                    "srun", "--exclusive", "--mem=64G", "-N", "1", "-n", "1", "-c", str(p),
                    exe, "-r", "5", "-p", str(p)
                ]
        else:
            cmd = [exe, "-r", "5", "-p", str(p)]
        # If we ended up using ./testInputs (not *_small), add -s for small
        if input_size == "small" and exe.endswith("testInputs"):
            cmd.append("-s")

        print("cmd: ", cmd)
        env = os.environ.copy()
        env["MIX"] = str(mix_value)
        if scheduler == "omp":
            env["OPENMP"] = "1"

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd_dir,                 # IMPORTANT: run from where the exe is
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                text=True,
                env=env,
            )
            with open(log_file, "w") as f:
                f.write(result.stdout)
            print("Captured output (first 500 chars):\n", result.stdout[:500])
        
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if e.stdout else ""
            with open(log_file, "w") as f:
                f.write(str(out))
            print(f"Timeout running {' '.join(cmd)} (cwd={cwd_dir}) with MIX={mix_value}, threads={p}")

        except subprocess.CalledProcessError as e:
            out = e.stdout or ""
            with open(log_file, "w") as f:
                f.write(str(out))
            print(f"Error running {' '.join(cmd)} (cwd={cwd_dir}) with MIX={mix_value}, threads={p}:")
            print(out[:1000])


def find_benchmark_entry(json_data, name):
    return next((item for item in json_data if item.get("name") == name), None)

def compute_mix_offset(generate_model: str) -> int:
    platform = generate_model or ""
    if platform not in INDEX_OFFSET_MAP.keys():
        platform = platform.split("-")[0]
    return INDEX_OFFSET_MAP.get(platform, 0)

def run_from_generated_outputs(args):
    """Original behavior: run evals from the generated outputs JSON."""
    with open(args.input, "r") as file:
        json_data = json.load(file)

    benchmarks_to_run = []
    if args.benchmark:
        target = find_benchmark_entry(json_data, args.benchmark)
        if target:
            benchmarks_to_run.append(target)
        else:
            print(f"Error: Benchmark '{args.benchmark}' not found in '{args.input}'.")
            return
    else:
        benchmarks_to_run = json_data

    for data in benchmarks_to_run:
        alg_dir = data["name"]
        outputs = data.get("outputs")
        if not outputs:
            print(f"No 'outputs' key found for '{alg_dir}', skipping.")
            continue

        generate_model = data["generate_model"]
        offset = compute_mix_offset(generate_model)

        alg_name = alg_dir.split("/", 1)[-1].replace("/", "_")
        destination_dir = os.path.join(args.output_dir, alg_dir)
        if not os.path.isdir(destination_dir):
            print(f"Directory does not exist: {destination_dir}, skipping.")
            continue

        run_make_cleanall(destination_dir)

        for jdx, _output in enumerate(outputs):
            mix_value = jdx + offset

            if args.dry:
                print(f"[DRY RUN] Would run in: {destination_dir}")
                print(f"[DRY RUN]   - Scheduler: {args.scheduler}, MIX: {mix_value}")
                print(f"[DRY RUN]   - testInputs threads: {args.threads}")
                continue

            if not run_make(destination_dir, mix_value, args.scheduler):
                print(f"Skipping test run for MIX={mix_value} due to compilation failure.")
                continue

            print(f"Running testInputs for {alg_name} with MIX={mix_value}")
            run_testInputs(
                    output_dir=args.output_dir,
                    alg_dir=alg_dir,
                    destination_dir=destination_dir,
                    mix_value=mix_value,
                    generate_model=generate_model,
                    alg_name=alg_name,
                    scheduler=args.scheduler,
                    server=args.server,
                    threads=args.threads,
                    input_size=args.input_size
                )


def run_from_rerun_requests(args):
    """
    New behavior: read rerun-requests.json and run only the requested evaluations.

    Notes:
      - For baseline (mix_value == "baseline"), we don't run make, we just run the testInputs binaries
        in the benchmark directory (assumes baseline binaries already exist / are built).
      - For non-baseline, we run `make MIX=<mix_value>` with the requested scheduler, then run testInputs.
      - Threads list comes from the rerun-requests.json entry, not args.threads (unless overridden).
    """
    with open(args.rerun_input, "r") as f:
        reruns = json.load(f)

    # Optional filter: only rerun one benchmark path
    if args.benchmark:
        reruns = [r for r in reruns if r.get("name") == args.benchmark]
        if not reruns:
            print(f"No rerun entries match benchmark '{args.benchmark}'.")
            return

    for r in reruns:
        alg_dir = r["name"]                      # e.g., nBody/parallelCK
        scheduler = r["scheduler"]               # omp/parlay/cilk
        server = r["server"]
        input_size = r.get("input_size", args.input_size)
        threads = r.get("threads", args.threads)

        generate_model = r.get("generate_model", "baseline")
        mix_value = r.get("mix_value", "baseline")

        alg_name = alg_dir.split("/", 1)[-1].replace("/", "_")
        destination_dir = os.path.join(args.output_dir, alg_dir)
        if not os.path.isdir(destination_dir):
            print(f"Directory does not exist: {destination_dir}, skipping.")
            continue

        # Allow overriding server/scheduler/input_size from CLI if desired
        if args.force_server:
            server = args.server
        if args.force_scheduler:
            scheduler = args.scheduler
        if args.force_input_size:
            input_size = args.input_size
        if args.force_threads:
            threads = args.threads

        print("=" * 80)
        print(f"RERUN: {alg_dir} | model={generate_model} mix={mix_value} "
              f"| scheduler={scheduler} | server={server} | input_size={input_size} | threads={threads}")

        if args.dry:
            print(f"[DRY RUN] Would rerun in: {destination_dir}")
            continue

        # Build step only if mix_value is not baseline
        if mix_value != "baseline":
            # You probably don't want cleanall for reruns (it can erase builds).
            # Keep it optional.
            if args.cleanall:
                run_make_cleanall(destination_dir)

            if not run_make(destination_dir, mix_value, scheduler):
                print(f"Skipping rerun due to compilation failure: MIX={mix_value}")
                continue

        # Run tests
        run_testInputs(
            output_dir=args.output_dir,
            alg_dir=alg_dir,
            destination_dir=destination_dir,
            mix_value=mix_value,
            generate_model=generate_model,
            alg_name=alg_name,
            scheduler=scheduler,
            server=server,
            threads=threads,
            input_size=input_size
        )


def main(args):
    if args.rerun_input:
        run_from_rerun_requests(args)
    else:
        run_from_generated_outputs(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations for ParlayLib benchmarks.")

    # Original mode inputs
    parser.add_argument("--input", "-i", type=str,
                        help="Path to the JSON file containing generated outputs (original mode).")

    # New rerun mode input
    parser.add_argument("--rerun-input", type=str, default=None,
                        help="Path to rerun-requests.json. If set, runs rerun mode.")

    parser.add_argument("--output_dir", "-o", type=str, default="../benchmarks",
                        help="Base directory where algorithm folders are located.")
    parser.add_argument("--benchmark", "-b", type=str, default=None,
                        help="Specific benchmark path to run (e.g., 'breadthFirstSearch/backForwardBFS').")

    parser.add_argument("--dry", action="store_true",
                        help="If provided, just print the commands without executing them.")
    parser.add_argument("-t", "--threads", nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 36, 48, 64], type=int,
                        help="Thread counts to use (original mode default; can override rerun mode with --force-threads).")
    parser.add_argument("-s", "--scheduler", choices=["parlay", "omp", "cilk"], required=True,
                        help="Scheduler to run (original mode default; can override rerun mode with --force-scheduler).")
    parser.add_argument("--server", required=True, type=str,
                        help="Server name for logs (original mode default; can override rerun mode with --force-server).")
    parser.add_argument("--input-size", type=str, choices=["small", "large"], default="large",
                        help="Input size selector (original mode default; can override rerun mode with --force-input-size).")

    # Rerun-mode override switches
    parser.add_argument("--force-threads", action="store_true",
                        help="In rerun mode, ignore threads in rerun-requests.json and use -t/--threads instead.")
    parser.add_argument("--force-scheduler", action="store_true",
                        help="In rerun mode, ignore scheduler in rerun-requests.json and use -s/--scheduler instead.")
    parser.add_argument("--force-server", action="store_true",
                        help="In rerun mode, ignore server in rerun-requests.json and use --server instead.")
    parser.add_argument("--force-input-size", action="store_true",
                        help="In rerun mode, ignore input_size in rerun-requests.json and use --input-size instead.")

    # Optional for rerun mode
    parser.add_argument("--cleanall", action="store_true",
                        help="In rerun mode, run 'make cleanall' before rebuilding non-baseline mixes.")

    args = parser.parse_args()

    # Validate: must provide --input in original mode
    if not args.rerun_input and not args.input:
        parser.error("Either provide --input (original mode) or --rerun-input (rerun mode).")

    main(args)
