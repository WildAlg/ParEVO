"""
Analyze iteration trends and solution speed from OpenEvolve results.

Usage:
    python analyze_trends.py <results_folder>

Reads: <results_folder>/<problem_id>/evaluation_results.csv
Outputs: pass_status.csv, speedup.csv
"""

import csv
import sys
from pathlib import Path


# Configuration
OUTPUT_DIR = Path(".")
PASS_STATUS_FILE = OUTPUT_DIR / "pass_status.csv"
SPEEDUP_FILE = OUTPUT_DIR / "speedup.csv"

ITERATION_THRESHOLDS = [1, 10, 30]
BASELINE_ITERATION_LIMIT = 3


def load_results(csv_path: Path) -> list[dict]:
    """Load evaluation results from CSV."""
    results = []
    
    if not csv_path.exists():
        return results
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'iteration': int(row['iteration']),
                'passed': row['passed'] == 'True',
                'avg_time': float(row['avg_time']),
            })
    
    return results


def analyze_problem(problem_id: str, results: list[dict]) -> tuple[dict, dict, str | None]:
    """
    Analyze a single problem's results.
    
    Returns:
        (pass_status_dict, speedup_dict, warning_message)
    """
    warning = None
    
    # Find baseline: first passing solution with smallest iteration number
    passing_results = [r for r in results if r['passed']]
    
    if not passing_results:
        # No passing solutions at all
        pass_status = {t: 0 for t in ITERATION_THRESHOLDS}
        speedup = {t: None for t in ITERATION_THRESHOLDS}
        return pass_status, speedup, warning
    
    # Find minimum iteration number among passing solutions
    baseline_result = min(passing_results, key=lambda r: r['iteration'])
    baseline_iteration = baseline_result['iteration']
    baseline_time = baseline_result['avg_time']
    
    if baseline_iteration > BASELINE_ITERATION_LIMIT:
        warning = f"Problem '{problem_id}': first passing solution at iteration {baseline_iteration} (> {BASELINE_ITERATION_LIMIT})"
    
    # Calculate metrics for each threshold
    pass_status = {}
    speedup = {}
    
    for threshold in ITERATION_THRESHOLDS:
        # Filter results within threshold
        threshold_results = [r for r in results if r['iteration'] <= threshold]
        threshold_passing = [r for r in threshold_results if r['passed']]
        
        # Pass status: 1 if at least one passing solution
        pass_status[threshold] = 1 if threshold_passing else 0
        
        # Speedup: baseline_time / min_avg_time among passing solutions
        if threshold_passing:
            min_time = min(r['avg_time'] for r in threshold_passing)
            if min_time > 0:
                speedup[threshold] = baseline_time / min_time
            else:
                speedup[threshold] = None
        else:
            speedup[threshold] = None
    
    return pass_status, speedup, warning


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_trends.py <results_folder>")
        sys.exit(1)
    
    results_folder = Path(sys.argv[1])
    
    if not results_folder.exists():
        print(f"Error: Folder not found: {results_folder}")
        sys.exit(1)
    
    # Find all problem folders
    problem_dirs = sorted([d for d in results_folder.iterdir() if d.is_dir()])
    
    if not problem_dirs:
        print(f"Error: No problem folders found in {results_folder}")
        sys.exit(1)
    
    print(f"Results folder: {results_folder}")
    print(f"Problems found: {len(problem_dirs)}")
    print()
    
    # Collect results
    all_pass_status = {}
    all_speedup = {}
    warnings = []
    
    for problem_dir in problem_dirs:
        problem_id = problem_dir.name
        csv_path = problem_dir / "evaluation_results.csv"
        
        if not csv_path.exists():
            print(f"Warning: No evaluation_results.csv for {problem_id}, skipping")
            continue
        
        results = load_results(csv_path)
        
        if not results:
            print(f"Warning: Empty results for {problem_id}, skipping")
            continue
        
        pass_status, speedup, warning = analyze_problem(problem_id, results)
        
        all_pass_status[problem_id] = pass_status
        all_speedup[problem_id] = speedup
        
        if warning:
            warnings.append(warning)
            print(f"WARNING: {warning}")
    
    if not all_pass_status:
        print("Error: No valid results found")
        sys.exit(1)
    
    # Print warnings summary
    if warnings:
        print()
        print(f"Total warnings: {len(warnings)}")
    
    # Calculate averages
    avg_pass_status = {}
    avg_speedup = {}
    
    for threshold in ITERATION_THRESHOLDS:
        # Average pass status
        values = [all_pass_status[p][threshold] for p in all_pass_status]
        avg_pass_status[threshold] = sum(values) / len(values)
        
        # Average speedup (only among valid values)
        speedup_values = [all_speedup[p][threshold] for p in all_speedup 
                         if all_speedup[p][threshold] is not None]
        if speedup_values:
            avg_speedup[threshold] = sum(speedup_values) / len(speedup_values)
        else:
            avg_speedup[threshold] = None
    
    # Write pass status CSV
    with open(PASS_STATUS_FILE, 'w', newline='') as f:
        fieldnames = ['problem_id'] + [f'iter_{t}' for t in ITERATION_THRESHOLDS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for problem_id in sorted(all_pass_status.keys()):
            row = {'problem_id': problem_id}
            for threshold in ITERATION_THRESHOLDS:
                row[f'iter_{threshold}'] = all_pass_status[problem_id][threshold]
            writer.writerow(row)
        
        # Average row
        row = {'problem_id': 'AVERAGE'}
        for threshold in ITERATION_THRESHOLDS:
            row[f'iter_{threshold}'] = f"{avg_pass_status[threshold]:.3f}"
        writer.writerow(row)
    
    print()
    print(f"Written: {PASS_STATUS_FILE}")
    
    # Write speedup CSV
    with open(SPEEDUP_FILE, 'w', newline='') as f:
        fieldnames = ['problem_id'] + [f'iter_{t}' for t in ITERATION_THRESHOLDS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for problem_id in sorted(all_speedup.keys()):
            row = {'problem_id': problem_id}
            for threshold in ITERATION_THRESHOLDS:
                val = all_speedup[problem_id][threshold]
                row[f'iter_{threshold}'] = f"{val:.3f}" if val is not None else ""
            writer.writerow(row)
        
        # Average row
        row = {'problem_id': 'AVERAGE'}
        for threshold in ITERATION_THRESHOLDS:
            val = avg_speedup[threshold]
            row[f'iter_{threshold}'] = f"{val:.3f}" if val is not None else ""
        writer.writerow(row)
    
    print(f"Written: {SPEEDUP_FILE}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()