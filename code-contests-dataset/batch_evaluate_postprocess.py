"""
Merge CSV results from different languages and compute speedup against sequential baseline.

Reads: batch_results_{cpp,rust}_t{threads}.csv
Writes: batch_results_t{threads}.csv (with speedup values)
"""

import csv
from pathlib import Path


# Configuration - modify as needed
THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64]
CSV_DIR = Path(__file__).parent / "batch_results"  # Directory containing the CSV files
LANGUAGES = ["cpp", "rust"]
SEQUENTIAL_SETUP = "sequential"


def load_csv(csv_path: Path) -> dict:
    """Load CSV into a dict: {(problem_id, setup): time_value}"""
    results = {}
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping")
        return results
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_id = row.get('problem_id', '')
            if not problem_id:
                continue
            for setup in row:
                if setup == 'problem_id':
                    continue
                val = row[setup].strip()
                if val != '':
                    try:
                        results[(problem_id, setup)] = float(val)
                    except ValueError:
                        results[(problem_id, setup)] = val
    return results


def compute_speedup(baseline_time, current_time):
    """
    Compute speedup = baseline / current.
    Returns None if either value is invalid.
    """
    # Check for invalid values
    if baseline_time is None or current_time is None:
        return None
    if isinstance(baseline_time, str) or isinstance(current_time, str):
        return None
    if baseline_time <= 0 or current_time <= 0:
        return None
    
    return baseline_time / current_time


def merge_and_compute_speedup(thread_count: int):
    """
    Merge cpp and rust CSVs for a given thread count and compute speedup.
    """
    # Load all language CSVs for this thread count
    merged_results = {}
    all_setups = set()
    all_problems = set()
    
    for lang in LANGUAGES:
        csv_path = CSV_DIR / f"batch_results_{lang}_t{thread_count}.csv"
        results = load_csv(csv_path)
        
        for (problem_id, setup), value in results.items():
            merged_results[(problem_id, setup)] = value
            all_setups.add(setup)
            all_problems.add(problem_id)
    
    if not merged_results:
        print(f"No data found for t{thread_count}, skipping")
        return
    
    # Get sequential baseline times
    sequential_times = {}
    for problem_id in all_problems:
        key = (problem_id, SEQUENTIAL_SETUP)
        if key in merged_results:
            val = merged_results[key]
            if isinstance(val, (int, float)) and val > 0:
                sequential_times[problem_id] = val
    
    # Compute speedups
    speedup_results = {}
    for (problem_id, setup), value in merged_results.items():
        baseline = sequential_times.get(problem_id)
        speedup = compute_speedup(baseline, value)
        
        if speedup is not None:
            speedup_results[(problem_id, setup)] = speedup
        elif isinstance(value, (int, float)) and value == -1:
            speedup_results[(problem_id, setup)] = "ERR"
        else:
            speedup_results[(problem_id, setup)] = ""
    
    # Sort setups and problems for consistent output
    sorted_setups = sorted(all_setups)
    sorted_problems = sorted(all_problems)
    
    # Write output CSV
    output_path = CSV_DIR / f"batch_results_t{thread_count}.csv"
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['problem_id'] + sorted_setups
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for problem_id in sorted_problems:
            row = {'problem_id': problem_id}
            for setup in sorted_setups:
                key = (problem_id, setup)
                if key in speedup_results:
                    val = speedup_results[key]
                    if isinstance(val, float):
                        row[setup] = f"{val:.3f}"
                    else:
                        row[setup] = val
                else:
                    row[setup] = ""
            writer.writerow(row)
    
    print(f"Written: {output_path} ({len(sorted_problems)} problems, {len(sorted_setups)} setups)")


def main():
    print(f"CSV Directory: {CSV_DIR}")
    print(f"Thread counts: {THREAD_COUNTS}")
    print(f"Languages: {LANGUAGES}")
    print(f"Sequential baseline: {SEQUENTIAL_SETUP}")
    print()
    
    for thread_count in THREAD_COUNTS:
        merge_and_compute_speedup(thread_count)
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()