"""
Batch evaluation script that maintains a CSV of results across different setups and problems.

Usage:
    python batch_evaluate.py <root_folder> [--csv results.csv] [--repeat 5] [--rerun]

Example:
    python batch_evaluate.py ./solutions --csv results.csv --repeat 5
"""

import os
import sys
import csv
import argparse
from pathlib import Path

# Import config to get the current LANGUAGE setting
from config import LANGUAGE

# Import the Judge class
from run_judge import Judge


# Hardcoded lists - modify as needed
PROBLEM_IDS = [
    'ccc15s1',
    'ccc17s3',
    'cco08p4',
    'coci11c1p3',
    'coci19c1p3',
    'coci21c2p1',
    'coci23c2p2',
    'joi21op1'
]

# Map setup name -> language ("cpp" or "rust")
SETUPS = {
    "gemini-2.5-pro-cpp": "cpp",
    "gemini-2.5-pro-parlay-cpp": "cpp",
    "qwen3-cpp": "cpp",
    "qwen3-cpp-dpo": "cpp",
    "qwen3-rust": "rust",
    "qwen3-rust-dpo": "rust",
    "sequential": "cpp",
}


def get_solution_path(root_folder: Path, setup: str, problem_id: str, language: str) -> Path:
    """Get the path to a solution file."""
    ext = "cpp" if language == "cpp" else "rs"
    return root_folder / setup / f"{problem_id}.{ext}"


def load_csv(csv_path: Path) -> dict:
    """Load existing CSV into a dict: {(problem_id, setup): avg_time}"""
    results = {}
    if csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                problem_id = row.get('problem_id', '')
                if not problem_id:
                    continue
                for setup in row:
                    if setup == 'problem_id':
                        continue
                    val = row[setup]
                    if val != '':
                        try:
                            results[(problem_id, setup)] = float(val)
                        except ValueError:
                            results[(problem_id, setup)] = val
    return results


def save_csv(csv_path: Path, results: dict, setups: list, problem_ids: list):
    """Save results dict to CSV."""
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['problem_id'] + setups
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for problem_id in problem_ids:
            row = {'problem_id': problem_id}
            for setup in setups:
                key = (problem_id, setup)
                if key in results:
                    row[setup] = results[key]
                else:
                    row[setup] = ''
            writer.writerow(row)


def evaluate_solution(problem_id: str, solution_path: Path, language: str, repeat: int) -> float:
    """
    Evaluate a solution and return avg_time.
    Returns -1 if error or wrong answer.
    Raises FileNotFoundError if solution doesn't exist.
    """
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution not found: {solution_path}")
    
    with open(solution_path, 'r') as f:
        source_code = f.read()
    
    judge = Judge(problem_id, time_limit=2, memory_limit=262144, language=language)
    results = judge.evaluate(source_code, repeat=repeat, display_result=False)
    
    if results.get('compile_error'):
        print(f"  ERROR: Compilation failed: {results['compile_error'][:200]}")
        return -1
    
    if not results.get('passed'):
        print(f"  ERROR: Wrong answer - {results.get('passed_tests', 0)}/{results.get('total_tests', 0)} tests passed")
        if results.get('feedback'):
            print(f"  Feedback: {results['feedback'][:200]}")
        return -1
    
    return results['avg_time']


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate solutions and maintain CSV results.')
    parser.add_argument('root_folder', type=str, help='Root folder containing setup subfolders')
    parser.add_argument('--csv', type=str, default='batch_results.csv', help='Path to CSV file (default: batch_results.csv)')
    parser.add_argument('--repeat', type=int, default=5, help='Number of repeat runs for timing (default: 5)')
    parser.add_argument('--rerun', action='store_true', help='Re-run all evaluations, ignoring existing results')
    
    args = parser.parse_args()
    
    root_folder = Path(args.root_folder)
    csv_path = Path(args.csv)
    repeat = args.repeat
    rerun = args.rerun
    
    if not root_folder.exists():
        print(f"Error: Root folder does not exist: {root_folder}")
        sys.exit(1)
    
    # Filter setups to only those matching current LANGUAGE
    active_setups = [setup for setup, lang in SETUPS.items() if lang == LANGUAGE]
    
    if not active_setups:
        print(f"No setups configured for language: {LANGUAGE}")
        sys.exit(1)
    
    print(f"Language: {LANGUAGE}")
    print(f"Active setups: {active_setups}")
    print(f"Problems: {len(PROBLEM_IDS)}")
    print(f"Repeat runs: {repeat}")
    print(f"CSV: {csv_path}")
    print()
    
    # Load existing results
    if rerun:
        results = {}
    else:
        results = load_csv(csv_path)
    
    # Process each setup and problem
    for setup in active_setups:
        setup_folder = root_folder / setup
        if not setup_folder.exists():
            print(f"Skipping setup '{setup}': folder not found")
            continue
        
        print(f"Processing setup: {setup}")
        
        for problem_id in PROBLEM_IDS:
            key = (problem_id, setup)
            
            # Skip if already evaluated (unless rerun)
            if key in results and not rerun:
                continue
            
            solution_path = get_solution_path(root_folder, setup, problem_id, LANGUAGE)
            
            if not solution_path.exists():
                # Skip silently if solution doesn't exist
                continue
            
            print(f"  Evaluating {problem_id}...", end=' ', flush=True)
            
            try:
                avg_time = evaluate_solution(problem_id, solution_path, LANGUAGE, repeat)
                results[key] = avg_time
                
                if avg_time >= 0:
                    print(f"{avg_time*1000:.2f}ms")
                # Error message already printed by evaluate_solution if -1
                
            except Exception as e:
                print(f"  ERROR: {e.with_traceback()}")
                results[key] = -1
            
            # Save after each evaluation for incremental progress
            save_csv(csv_path, results, active_setups, PROBLEM_IDS)
    
    print()
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()