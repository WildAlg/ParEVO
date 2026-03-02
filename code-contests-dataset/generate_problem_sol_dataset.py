#!/usr/bin/env python
"""
Generate solution dataset including passing status and ERROR messages.
FILTERS out infrastructure/permission errors.
EXCLUDES specified test problems.

Creates a unified JSONL dataset where each entry contains:
- problem_description
- solution (code)
- runtime
- passed (boolean)
- error (compiler/runtime error message, if any)
"""

import csv
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Set

def get_available_problems(results_dir: Path) -> List[str]:
    """Get list of available problem IDs from results directory"""
    if not results_dir.exists():
        return []
    return [
        d.name for d in results_dir.iterdir() 
        if d.is_dir() and (d / "evaluation_results.csv").exists()
    ]

def load_problem_description(problem_dir: Path) -> str:
    """Load the problem description text file"""
    desc_path = problem_dir / "problem_description.txt"
    try:
        if desc_path.exists():
            return desc_path.read_text(encoding='utf-8').strip()
    except Exception:
        pass
    return ""

def is_infrastructure_error(error_msg: str) -> bool:
    """
    Check if the error is related to system/infrastructure issues 
    rather than code logic.
    """
    if not error_msg:
        return False
        
    error_lower = error_msg.lower()
    
    # 1. Check for specific forbidden patterns
    if "cannot open" in error_lower:
        return True
    if "permission denied" in error_lower:
        return True
        
    # 2. Check for the specific cargo path (exact match or prefix)
    cargo_path = "/apps/software/2024a/software/Rust/1.83.0-GCCcore-13.3.0/bin/cargo:"
    if error_msg.strip() == cargo_path:
        return True
    
    if error_msg.strip().startswith(cargo_path):
        return True

    return False

def load_csv_results(csv_path: Path, problem_id: str) -> List[Dict]:
    """Load and parse CSV results for a specific problem"""
    results = []
    if not csv_path.exists():
        return results

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                passed = row.get('passed', 'false').lower() == 'true'
                
                try:
                    avg_time = float(row.get('avg_time', 0))
                except (ValueError, TypeError):
                    avg_time = -1.0

                error_msg = (
                    row.get('error') or 
                    row.get('stderr') or 
                    row.get('message') or 
                    row.get('compile_error') or 
                    ""
                ).strip()

                if is_infrastructure_error(error_msg):
                    continue

                results.append({
                    'problem_id': problem_id,
                    'code': row['code'],
                    'runtime': avg_time,
                    'passed': passed,
                    'error': error_msg,
                })
            except (ValueError, KeyError):
                continue
    
    return results

def deduplicate_solutions(results: List[Dict]) -> List[Dict]:
    """
    Remove duplicates.
    Prioritize: passed > failed, lower runtime, presence of error message (if failed).
    """
    if not results:
        return []
    
    code_to_best = {}
    
    for result in results:
        code = result['code']
        
        if code not in code_to_best:
            code_to_best[code] = result
        else:
            existing = code_to_best[code]
            new_is_better = False
            
            if result['passed'] and not existing['passed']:
                new_is_better = True
            elif result['passed'] == existing['passed']:
                if result['passed']:
                    if result['runtime'] < existing['runtime']:
                        new_is_better = True
                else:
                    if len(result['error']) > len(existing['error']):
                        new_is_better = True
            
            if new_is_better:
                code_to_best[code] = result
    
    return list(code_to_best.values())

def generate_dataset_entry(problem_desc: str, result: Dict) -> Dict:
    """Generate the JSON entry including error info"""
    return {
        "problem_description": problem_desc,
        "solution": result['code'],
        "runtime": result['runtime'],
        "passed": result['passed'],
        "error": result['error'],
        "metadata": {
            "problem_id": result['problem_id'],
            "runtime_ms": result['runtime'] * 1000
        }
    }

def parse_exclusions(exclusion_arg: str) -> Set[str]:
    """Parse comma-separated string or file path into a set of IDs"""
    if not exclusion_arg:
        return set()
    
    path = Path(exclusion_arg)
    if path.is_file():
        # Read from file (one ID per line)
        print(f"Loading exclusions from file: {path}")
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    else:
        # Parse comma-separated string
        return {pid.strip() for pid in exclusion_arg.split(',') if pid.strip()}

def main():
    parser = argparse.ArgumentParser(description="Generate clean dataset with solutions and errors")
    parser.add_argument("--results-dir", type=str, default="results", 
                       help="Root directory containing problem subdirectories")
    parser.add_argument("--output", type=str, default="contrastive_error_dataset.jsonl", 
                       help="Output JSONL file path")
    parser.add_argument("--exclude-problems", type=str, default="", 
                       help="Comma-separated list of problem IDs to skip, OR path to a .txt file with IDs")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    problems = get_available_problems(results_dir)
    excluded_ids = parse_exclusions(args.exclude_problems)
    
    if excluded_ids:
        print(f"Excluding {len(excluded_ids)} problems from dataset generation.")
    
    print(f"Found {len(problems)} total problems.")
    
    dataset = []
    stats = {"passed": 0, "failed_with_error": 0, "failed_silent": 0, "excluded": 0}
    example_errors: Set[str] = set()

    for problem_id in problems:
        # SKIP if in exclusion list
        if problem_id in excluded_ids:
            stats["excluded"] += 1
            continue

        problem_dir = results_dir / problem_id
        
        description = load_problem_description(problem_dir)
        if not description:
            continue

        csv_path = problem_dir / "evaluation_results.csv"
        results = load_csv_results(csv_path, problem_id)
        unique_results = deduplicate_solutions(results)
        
        for result in unique_results:
            dataset.append(generate_dataset_entry(description, result))
            
            if result['passed']:
                stats["passed"] += 1
            elif result['error']:
                stats["failed_with_error"] += 1
                clean_err = result['error'][:2000].replace('\n', ' ')
                example_errors.add(clean_err)
            else:
                stats["failed_silent"] += 1

        if len(unique_results) > 0:
            print(f"Processed {problem_id}: {len(unique_results)} solutions")

    # Output
    print(f"\n=== Summary ===")
    print(f"Total entries: {len(dataset)}")
    print(f"  - Passed: {stats['passed']}")
    print(f"  - Failed (with error msg): {stats['failed_with_error']}")
    print(f"  - Failed (no error msg): {stats['failed_silent']}")
    print(f"  - Problems Excluded (Test Set): {stats['excluded']}")
    
    if example_errors:
        print(f"\n=== Example Error Messages (Random 5) ===")
        samples = list(example_errors)
        random.shuffle(samples)
        for i, err in enumerate(samples[:5]):
            print(f"{i+1}. {err}...")
            
    output_path = Path(args.output)
    print(f"\nWriting to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print("Done.")

if __name__ == "__main__":
    main()