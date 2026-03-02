#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
from collections import defaultdict

def safe_run(cmd: str, cwd: str):
    """
    Run command in cwd. Return (ok, stdout, stderr, returncode).
    Never raises.
    """
    try:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        return (r.returncode == 0), (r.stdout or ""), (r.stderr or ""), r.returncode
    except FileNotFoundError as e:
        return False, "", f"FileNotFoundError: {e}", 127
    except Exception as e:
        return False, "", f"Exception: {e}", 1

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def scheduler_to_make_flag(scheduler: str) -> str:
    s = (scheduler or "").lower()
    if s in ("parlay", "parlaylib"):
        return ""              # plain `make`
    if s in ("omp", "openmp"):
        return "OPENMP=1"
    raise ValueError(f"Unknown scheduler: {scheduler!r} (expected parlay or omp)")

def input_size_to_exec(input_size: str) -> str:
    s = (input_size or "").lower()
    if s == "large":
        return "./testInputs"
    if s == "small":
        return "./testInputs_small"
    raise ValueError(f"Unknown input_size: {input_size!r} (expected large or small)")

def log_name(alg_dir: str, scheduler: str, server: str, input_size: str, threads: int) -> str:
    # alg_dir in JSON looks like: "rayCast/kdTree"
    alg_name = alg_dir.split("/", 1)[1] if "/" in alg_dir else alg_dir
    sched_tag = "parlay" if scheduler.lower() in ("parlay", "parlaylib") else "omp"
    return f"{alg_name}_{sched_tag}_{server}_{input_size}_{threads}.log"

def rerun_case(base_dir: str, entry: dict, rounds: int, clean_between: bool):
    """
    Rerun a single JSON entry (may contain multiple threads).
    Builds once per entry (algorithm + scheduler), then runs requested threads.
    """
    alg_dir = entry["name"]
    scheduler = entry.get("scheduler", "parlay")
    server = entry.get("server", "unknown")
    input_size = entry.get("input_size", "large")
    threads_list = entry.get("threads", [])

    alg_path = os.path.join(base_dir, alg_dir)
    if not os.path.isdir(alg_path):
        print(f"[SKIP] Algorithm dir not found: {alg_path}")
        return

    baseline_runtime = os.path.join(alg_path, "baseline_runtime")
    ensure_dir(baseline_runtime)

    make_flag = scheduler_to_make_flag(scheduler)
    exec_path = input_size_to_exec(input_size)

    if clean_between:
        ok, out, err, rc = safe_run("make cleanall", cwd=alg_path)
        if not ok:
            print(f"[WARN] make cleanall failed in {alg_dir} (rc={rc})\n{err}")

    # Build once for this entry
    build_cmd = "make" if make_flag == "" else f"make {make_flag}"
    ok, out, err, rc = safe_run(build_cmd, cwd=alg_path)
    if not ok:
        print(f"[ERROR] Build failed for {alg_dir} ({scheduler}) rc={rc}")
        # still write a build-failure marker for each requested thread, so you can track it
        for t in threads_list:
            log_path = os.path.join(baseline_runtime, log_name(alg_dir, scheduler, server, input_size, t))
            with open(log_path, "a") as f:
                f.write("\n" + "="*80 + "\n")
                f.write("RERUN RESULT: BUILD FAILED\n")
                f.write(f"cmd: {build_cmd}\n")
                f.write(f"rc: {rc}\n\n")
                f.write("stdout:\n" + out + "\n")
                f.write("stderr:\n" + err + "\n")
        return

    # Run threads
    for t in threads_list:
        run_cmd = f"{exec_path} -r {rounds} -p {t}"
        ok2, out2, err2, rc2 = safe_run(run_cmd, cwd=alg_path)

        log_path = os.path.join(baseline_runtime, log_name(alg_dir, scheduler, server, input_size, t))
        with open(log_path, "a") as f:
            f.write("\n" + "="*80 + "\n")
            f.write("RERUN RESULT\n")
            f.write(f"algorithm: {alg_dir}\n")
            f.write(f"scheduler: {scheduler}\n")
            f.write(f"server: {server}\n")
            f.write(f"input_size: {input_size}\n")
            f.write(f"threads: {t}\n")
            f.write(f"rounds: {rounds}\n")
            if entry.get("reason"):
                f.write(f"original_reason: {entry.get('reason')}\n")
            f.write("\n")
            f.write(f"build_cmd: {build_cmd}\n")
            f.write("build_stdout:\n" + out + "\n")
            f.write("build_stderr:\n" + err + "\n")
            f.write("\n")
            f.write(f"run_cmd: {run_cmd}\n")
            f.write(f"run_rc: {rc2}\n")
            f.write("run_stdout:\n" + out2 + "\n")
            f.write("run_stderr:\n" + err2 + "\n")

        status = "OK" if ok2 else "FAIL"
        print(f"[{status}] {alg_dir} {scheduler} {input_size} t={t} (rc={rc2}) -> {log_path}")

def main():
    ap = argparse.ArgumentParser(description="Rerun failed baseline benchmarks based on rerun-requests.json.")
    ap.add_argument("--requests-json", "-i", required=True, help="Path to rerun-requests.json")
    ap.add_argument("--base-dir", "-b", default="../benchmarks", help="Base directory containing algorithm folders")
    ap.add_argument("--rounds", "-r", type=int, default=5, help="Rounds to pass to testInputs")
    ap.add_argument("--clean-between", action="store_true",
                    help="Run `make cleanall` before each entry build (slower but cleaner).")
    args = ap.parse_args()

    try:
        with open(args.requests_json, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print("Expected requests-json to be a LIST of objects.")
        sys.exit(1)

    # Optional: group by (alg_dir, scheduler, server, input_size) to merge threads,
    # so we build once per unique group even if JSON has duplicates.
    grouped = {}
    for entry in data:
        try:
            key = (
                entry["name"],
                entry.get("scheduler", "parlay"),
                entry.get("server", "unknown"),
                entry.get("input_size", "large"),
            )
        except KeyError:
            print("[SKIP] entry missing required key 'name'")
            continue

        if key not in grouped:
            grouped[key] = dict(entry)
            grouped[key]["threads"] = list(entry.get("threads", []))
        else:
            grouped[key]["threads"].extend(entry.get("threads", []))

    # Deduplicate threads per group
    for key, entry in grouped.items():
        entry["threads"] = sorted(set(int(x) for x in entry.get("threads", [])))

    print(f"Loaded {len(data)} request objects -> {len(grouped)} unique rerun groups.")
    for _, entry in grouped.items():
        rerun_case(args.base_dir, entry, rounds=args.rounds, clean_between=args.clean_between)

    print("Done.")

if __name__ == "__main__":
    main()
