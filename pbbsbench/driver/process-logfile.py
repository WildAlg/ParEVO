# process-logfile.py (modified)
import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

# Copied from run-output.py for consistency
INDEX_OFFSET_MAP = {
    "claude": 20,
    "gpt": 40,
    "gemini": 60,
    "deepseek": 80,
    "gemini-2.5-pro-finetuned": 200
}

ABNORMAL_MARKER = "TEST TERMINATED ABNORMALLY"


def parse_log_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse log filename into metadata.

    Supported formats:

      1) {model}_{mix}_{alg}_{scheduler}_{server}_{input_size}_{threads}.txt|.log
         gemini-2.5-pro_62_backForwardBFS_omp_bouchet_large_8.txt

      2) {model}_{mix}_{alg}_{scheduler}_{server}_{threads}.txt|.log
         gemini-2.5-pro_62_backForwardBFS_omp_bouchet_8.txt

      3) {alg}_{scheduler}_{server}_{input_size}_{threads}.txt|.log   (baseline)
         parallelCK_omp_bouchet_small_16.log

      4) {alg}_{scheduler}_{server}_{threads}.txt|.log               (baseline, no input_size)
         parallelCK_omp_bouchet_16.log
    """
    stem, ext = os.path.splitext(filename)
    if ext not in (".txt", ".log"):
        return None

    parts = stem.split("_")

    try:
        # (1) 7-part, with model+mix+input_size
        if len(parts) == 7:
            model, mix_s, alg, scheduler, server, input_size, threads_s = parts
            return {
                "model": model,
                "mix_value": int(mix_s),
                "algorithm": alg,
                "scheduler": scheduler,
                "server": server,
                "input_size": input_size,
                "threads": int(threads_s),
            }

        # (2) 6-part, with model+mix, no input_size
        if len(parts) == 6:
            model, mix_s, alg, scheduler, server, threads_s = parts
            return {
                "model": model,
                "mix_value": int(mix_s),
                "algorithm": alg,
                "scheduler": scheduler,
                "server": server,
                "input_size": None,
                "threads": int(threads_s),
            }

        # (3) 5-part baseline, with input_size
        if len(parts) == 5:
            alg, scheduler, server, input_size, threads_s = parts
            return {
                "model": "baseline",
                "mix_value": "baseline",
                "algorithm": alg,
                "scheduler": scheduler,
                "server": server,
                "input_size": input_size,
                "threads": int(threads_s),
            }

        # (4) 4-part baseline, no input_size
        if len(parts) == 4:
            alg, scheduler, server, threads_s = parts
            return {
                "model": "baseline",
                "mix_value": "baseline",
                "algorithm": alg,
                "scheduler": scheduler,
                "server": server,
                "input_size": None,
                "threads": int(threads_s),
            }

    except ValueError:
        return None

    return None



def platform_from_model(model: str) -> str:
    # Mirrors run-output.py behavior: generate_model.split("-")[0]
    return (model or "").split("-")[0]


def process_log_file(log_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single log file to extract performance data (geomeans + raw runtimes),
    and also detect abnormal termination.

    Returns:
      dict with metadata + parsed results; includes:
        - abnormal_terminated: bool
        - abnormal_marker: str (if present)
    """
    filename = os.path.basename(log_file_path)
    meta = parse_log_filename(filename)
    if meta is None:
        print(f"Error: Unexpected filename format {filename}. Skipping.")
        return None

    parsed_data: Dict[str, Any] = {
        "model": meta["model"],
        "mix_value": meta["mix_value"],
        "algorithm": meta["algorithm"],
        "scheduler": meta["scheduler"],
        "server": meta["server"],
        "input_size": meta["input_size"],
        "threads": meta["threads"],
        "individual_geomeans": {},
        "overall_geomean": None,
        "raw_runtimes": {},
        "error": None,
        "abnormal_terminated": False,
        "abnormal_marker": None,
        "log_file": log_file_path,
    }

    individual_run_regex = re.compile(r"(.+)\s+:\s+.*?:\s+(.*?),\s+geomean\s*=\s*(\S+)")
    overall_geomean_regex = re.compile(r"geomean of geomeans\s+=\s+(\S+)")
    error_regex = re.compile(r"RuntimeError: (.*)")

    with open(log_file_path, "r") as f:
        log_content = f.read()
        lines = log_content.splitlines()

    # Detect abnormal termination marker
    if ABNORMAL_MARKER in log_content:
        parsed_data["abnormal_terminated"] = True
        parsed_data["abnormal_marker"] = ABNORMAL_MARKER
        # store a helpful error string too
        parsed_data["error"] = ABNORMAL_MARKER

    for line in lines:
        match_individual = individual_run_regex.search(line)
        if match_individual:
            graph_name = match_individual.group(1).strip()
            runtimes_str = match_individual.group(2)
            geomean = float(match_individual.group(3))

            parsed_data["individual_geomeans"][graph_name] = geomean
            runtimes = [float(s) for s in re.findall(r"\'([\d\.]+)\'", runtimes_str)]
            parsed_data["raw_runtimes"][graph_name] = runtimes
            continue

        match_overall = overall_geomean_regex.search(line)
        if match_overall:
            parsed_data["overall_geomean"] = float(match_overall.group(1))
            continue

    # Check for RuntimeError lines
    error_match = error_regex.search(log_content)
    if error_match:
        parsed_data["error"] = error_match.group(1).strip()

    # If no overall geomean and no error, mark as parse error (but preserve abnormal marker if set)
    if parsed_data["overall_geomean"] is None and parsed_data["error"] is None:
        parsed_data["error"] = "Could not find overall geomean or an error message."

    return parsed_data


def add_rerun_request(
    rerun_requests: Dict[Tuple[Any, ...], Dict[str, Any]],
    info: Dict[str, Any],
    alg_dir: str,
) -> None:
    """
    Record enough info to rerun this log.

    We'll group reruns by (alg_dir, scheduler, server, model, mix_value, input_size),
    and store the list of thread counts that need rerun.
    """
    key = (
        alg_dir,
        info["scheduler"],
        info["server"],
        info["model"],
        info["mix_value"],
        info["input_size"],
    )
    if key not in rerun_requests:
        rerun_requests[key] = {
            "name": alg_dir,  # matches run-output.py "name" (relative algorithm folder)
            "generate_model": info["model"],
            "mix_value": info["mix_value"],
            "algorithm": info["algorithm"],
            "scheduler": info["scheduler"],
            "server": info["server"],
            "input_size": info["input_size"],
            "threads": [],
            "reason": ABNORMAL_MARKER,
            "log_files": [],
        }

    t = int(info["threads"])
    if t not in rerun_requests[key]["threads"]:
        rerun_requests[key]["threads"].append(t)

    rerun_requests[key]["log_files"].append(info.get("log_file"))


def main():
    parser = argparse.ArgumentParser(description="Process benchmark log files from a JSON input file.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the JSON file containing generated outputs (same one used by run-output.py).")
    parser.add_argument("--base_dir", "-b", type=str, default="../benchmarks",
                        help="Base directory where algorithm folders are located.")
    parser.add_argument("--output_file", "-o", type=str, default="processed_benchmark_data.json",
                        help="Output JSON file for parsed performance data.")
    parser.add_argument("--log_dir_name", type=str, default="eval-output",
                        help="Name of folder containing the log files.")
    parser.add_argument(
        "--log_file_formats", nargs="+", default=[".txt", ".log"], help="Log file extensions to include (e.g., .txt .log).",)
    parser.add_argument("--rerun_output", type=str, default="rerun-requests.json",
                        help="Output JSON file listing abnormal runs to rerun.")
    args = parser.parse_args()

    with open(args.input, "r") as file:
        json_data = json.load(file)

    combined_data_dict: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    rerun_requests: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for data in json_data:
        try:
            alg_dir = data["name"]              # e.g. breadthFirstSearch/backForwardBFS
            _generate_model = data["generate_model"]
        except KeyError:
            print("Skipping a JSON entry due to missing 'name' or 'generate_model' key.")
            continue

        log_directory = os.path.join(args.base_dir, alg_dir, args.log_dir_name)
        if not os.path.isdir(log_directory):
            print(f"Error: The directory '{log_directory}' does not exist. Skipping.")
            continue

        log_files = [
            f for f in os.listdir(log_directory)
            if any(f.endswith(ext) for ext in args.log_file_formats)
        ]
        if not log_files:
            print(f"No {args.log_file_formats} log files found in '{log_directory}'. Skipping.")
            continue

        print(f"Found {len(log_files)} log files in '{log_directory}'. Processing...")

        for log_file in log_files:
            log_file_path = os.path.join(log_directory, log_file)
            try:
                processed_info = process_log_file(log_file_path)
                if not processed_info:
                    continue

                # If abnormal termination, record rerun info (and still store parsed partial data)
                if processed_info.get("abnormal_terminated"):
                    add_rerun_request(rerun_requests, processed_info, alg_dir)

                # group key (threads excluded)
                key = (
                    processed_info["model"],
                    processed_info["mix_value"],
                    processed_info["algorithm"],
                    processed_info["scheduler"],
                    processed_info["server"],
                    processed_info["input_size"],
                )

                if key not in combined_data_dict:
                    combined_data_dict[key] = {
                        "model": processed_info["model"],
                        "mix_value": processed_info["mix_value"],
                        "algorithm": processed_info["algorithm"],
                        "scheduler": processed_info["scheduler"],
                        "server": processed_info["server"],
                        "input_size": processed_info["input_size"],
                        "results_by_threads": {},
                    }

                thread_key = str(processed_info["threads"])
                combined_data_dict[key]["results_by_threads"][thread_key] = {
                    "threads": processed_info["threads"],
                    "individual_geomeans": processed_info["individual_geomeans"],
                    "overall_geomean": processed_info["overall_geomean"],
                    "raw_runtimes": processed_info["raw_runtimes"],
                    "error": processed_info["error"],
                    "abnormal_terminated": processed_info["abnormal_terminated"],
                    "log_file": processed_info["log_file"],
                }

            except Exception as e:
                print(f"An unexpected error occurred while processing {log_file}: {e}. Skipping.")
                continue

    all_data = list(combined_data_dict.values())
    print(f"Successfully processed {len(all_data)} unique benchmarks. Saving data...")

    with open(args.output_file, "w") as f:
        json.dump(all_data, f, indent=4)

    # Write rerun requests (sorted thread lists)
    rerun_list = list(rerun_requests.values())
    for r in rerun_list:
        r["threads"] = sorted(r["threads"])

    with open(args.rerun_output, "w") as f:
        json.dump(rerun_list, f, indent=4)

    print(f"Data saved to {args.output_file}")
    print(f"Rerun list saved to {args.rerun_output} (count={len(rerun_list)})")


if __name__ == "__main__":
    main()
