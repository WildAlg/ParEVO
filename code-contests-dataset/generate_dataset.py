#!/usr/bin/env python
"""
Generate performance comparison dataset from ALL OpenEvolve CSV results.

Creates a unified JSONL dataset with pairs of slow/fast code across all problems
for teaching LLMs to write performant code.
"""

import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict
import random


def get_available_problems(results_dir: Path) -> List[str]:
    """Get list of available problem IDs from results directory"""
    if not results_dir.exists():
        return []
    return [
        d.name for d in results_dir.iterdir() 
        if d.is_dir() and (d / "evaluation_results.csv").exists()
    ]


def load_csv_results(csv_path: Path, problem_id: str) -> List[Dict]:
    """Load and parse CSV results for a specific problem"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse fields
            try:
                passed = row['passed'].lower() == 'true'
                combined_score = float(row['combined_score'])
                avg_time = float(row['avg_time'])
                
                # Skip failed solutions or zero score
                if not passed or combined_score == 0.0:
                    continue
                
                results.append({
                    'problem_id': problem_id,
                    'iteration': int(row['iteration']),
                    'code': row['code'],
                    'avg_time': avg_time,
                    'combined_score': combined_score,
                    'tests_passed': int(row['tests_passed']),
                    'tests_total': int(row['tests_total']),
                })
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row in {problem_id} due to error: {e}")
                continue
    
    return results


def deduplicate_solutions(results: List[Dict]) -> List[Dict]:
    """Remove duplicate code, keeping the one with best performance"""
    code_to_best = {}
    
    for result in results:
        code = result['code']
        if code not in code_to_best or result['avg_time'] < code_to_best[code]['avg_time']:
            code_to_best[code] = result
    
    return list(code_to_best.values())


def generate_pairs(results: List[Dict], min_speedup_ratio: float = 1.2) -> List[Dict]:
    """
    Generate pairs of (slow, fast) code.
    
    Strategy:
    1. Sort solutions by avg_time (fastest first)
    2. Always include the fastest solution in at least one pair
    3. Generate pairs between adjacent solutions if speedup >= min_speedup_ratio
    4. Also generate pairs between fastest and all slower solutions
    """
    # Sort by avg_time (fastest first)
    sorted_results = sorted(results, key=lambda x: x['avg_time'])
    
    if len(sorted_results) < 2:
        return []
    
    pairs = []
    fastest = sorted_results[0]
    
    # Strategy 1: Adjacent pairs (greedily pick distinct performance levels)
    for i in range(len(sorted_results) - 1):
        fast = sorted_results[i]
        slow = sorted_results[i + 1]
        
        speedup_ratio = slow['avg_time'] / fast['avg_time']
        
        if speedup_ratio >= min_speedup_ratio:
            pairs.append({
                'problem_id': fast['problem_id'],
                'fast': fast,
                'slow': slow,
                'speedup_ratio': speedup_ratio,
            })
    
    # Strategy 2: Ensure fastest solution appears in pairs with significantly slower ones
    # Pick solutions at different performance tiers
    for i in range(1, len(sorted_results)):
        slow = sorted_results[i]
        speedup_ratio = slow['avg_time'] / fastest['avg_time']
        
        # Include if significantly slower and not already covered by adjacent pairs
        if speedup_ratio >= min_speedup_ratio * 1.5:  # Higher threshold for non-adjacent
            # Check if this pair is novel (not just adjacent pair)
            is_novel = True
            for existing_pair in pairs:
                if (existing_pair['fast']['iteration'] == fastest['iteration'] and 
                    existing_pair['slow']['iteration'] == slow['iteration']):
                    is_novel = False
                    break
            
            if is_novel:
                pairs.append({
                    'problem_id': fastest['problem_id'],
                    'fast': fastest,
                    'slow': slow,
                    'speedup_ratio': speedup_ratio,
                })
    
    return pairs


def format_code_block(code: str, label: str) -> str:
    """Format code with clear label"""
    return f"=== {label} ===\n{code.strip()}\n"


def generate_instruction_variants() -> List[str]:
    """Generate diverse instruction prompts"""
    return [
        "Identify which code implementation runs faster.",
        "Determine which of the two code solutions has better performance.",
        "Compare the runtime efficiency of these two implementations and identify the faster one.",
        "Which code snippet will execute more quickly?",
        "Analyze these two code solutions and select the more performant one.",
        "Identify the more efficient implementation from the two options below.",
        "Select the code with superior execution speed.",
    ]


def generate_dataset_entry(pair: Dict, instruction: str, include_explanation: bool = False) -> Dict:
    """Generate a single dataset entry from a code pair"""
    fast_code = pair['fast']['code']
    slow_code = pair['slow']['code']
    speedup_ratio = pair['speedup_ratio']
    fast_time = pair['fast']['avg_time']
    slow_time = pair['slow']['avg_time']
    
    # Randomly shuffle order of presentation
    if random.random() < 0.5:
        input_text = (
            format_code_block(fast_code, "Code A") +
            "\n" +
            format_code_block(slow_code, "Code B")
        )
        answer_label = "Code A"
    else:
        input_text = (
            format_code_block(slow_code, "Code A") +
            "\n" +
            format_code_block(fast_code, "Code B")
        )
        answer_label = "Code B"
    
    # Generate output
    if include_explanation:
        output_text = (
            f"{answer_label} is faster.\n\n"
            f"Performance comparison:\n"
            # f"- {answer_label}: {fast_time*1000:.2f}ms average runtime\n"
            # f"- {'Code B' if answer_label == 'Code A' else 'Code A'}: {slow_time*1000:.2f}ms average runtime\n"
            f"- Speedup: {speedup_ratio:.2f}x faster"
        )
    else:
        output_text = f"{answer_label}"
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": {
            "problem_id": pair['problem_id'],
            "speedup_ratio": speedup_ratio,
            "fast_time_ms": fast_time * 1000,
            "slow_time_ms": slow_time * 1000,
        }
    }

def deduplicate_solutions(results: List[Dict], time_similarity_threshold: float = 0.05) -> List[Dict]:
    """
    Remove duplicate code AND solutions with very similar performance.
    
    Args:
        results: List of solution results
        time_similarity_threshold: Consider solutions with runtime within this fraction as duplicates
                                   (e.g., 0.05 = 5% difference)
    
    Returns:
        List of unique, performance-distinct solutions
    """
    if not results:
        return []
    
    # First pass: remove exact code duplicates, keeping best performance
    code_to_best = {}
    for result in results:
        code = result['code']
        if code not in code_to_best or result['avg_time'] < code_to_best[code]['avg_time']:
            code_to_best[code] = result
    
    unique_by_code = list(code_to_best.values())
    
    # Second pass: remove solutions with very similar runtime
    # Sort by avg_time
    sorted_solutions = sorted(unique_by_code, key=lambda x: x['avg_time'])
    
    filtered = []
    for solution in sorted_solutions:
        # Check if this solution has significantly different performance from already selected ones
        is_distinct = True
        for existing in filtered:
            # Calculate relative difference
            time_diff = abs(solution['avg_time'] - existing['avg_time'])
            relative_diff = time_diff / existing['avg_time']
            
            if relative_diff < time_similarity_threshold:
                # Too similar in performance, likely duplicate algorithm with minor variations
                is_distinct = False
                break
        
        if is_distinct:
            filtered.append(solution)
    
    return filtered

def main():
    parser = argparse.ArgumentParser(description="Generate performance comparison dataset from all CSVs")
    parser.add_argument("--results-dir", type=str, default="results", 
                       help="Results directory containing problem subdirs (default: results)")
    parser.add_argument("--output", type=str, default="performance_dataset.jsonl", 
                       help="Output JSONL file")
    parser.add_argument("--min-speedup", type=float, default=1.2, 
                       help="Minimum speedup ratio to include pair (default: 1.2 = 20%% faster)")
    parser.add_argument("--time-similarity", type=float, default=0.05,
                       help="Solutions with runtime within this fraction are considered duplicates (default: 0.05 = 5%%)")
    parser.add_argument("--include-explanation", action="store_true",
                       help="Include performance metrics in output")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Get all available problems
    problems = get_available_problems(results_dir)
    print(f"Found {len(problems)} problems: {problems}")
    
    if len(problems) == 0:
        print("Error: No problems found in results directory")
        return
    
    # Load all results
    all_pairs = []
    problem_stats = {}
    
    for problem_id in problems:
        csv_path = results_dir / problem_id / "evaluation_results.csv"
        print(f"\n=== Processing {problem_id} ===")
        
        results = load_csv_results(csv_path, problem_id)
        print(f"  Loaded {len(results)} valid solutions")
        
        if len(results) < 2:
            print(f"  Skipping {problem_id}: need at least 2 valid solutions")
            continue
        
        unique_results = deduplicate_solutions(results, time_similarity_threshold=args.time_similarity)
        print(f"  After deduplication: {len(unique_results)} performance-distinct solutions")
        
        if len(unique_results) < 2:
            print(f"  Skipping {problem_id}: all solutions have similar performance")
            continue
        
        pairs = generate_pairs(unique_results, min_speedup_ratio=args.min_speedup)
        print(f"  Generated {len(pairs)} pairs")
        
        if len(pairs) > 0:
            speedups = [p['speedup_ratio'] for p in pairs]
            print(f"  Speedup range: {min(speedups):.2f}x - {max(speedups):.2f}x (avg: {sum(speedups)/len(speedups):.2f}x)")
            
            # Check if fastest solution is included
            fastest_iter = min(unique_results, key=lambda x: x['avg_time'])['iteration']
            fastest_in_pairs = any(
                p['fast']['iteration'] == fastest_iter or p['slow']['iteration'] == fastest_iter 
                for p in pairs
            )
            print(f"  Fastest solution included: {fastest_in_pairs}")
            
            problem_stats[problem_id] = {
                'pairs': len(pairs),
                'unique_solutions': len(unique_results),
                'speedup_range': (min(speedups), max(speedups)),
            }
        
        all_pairs.extend(pairs)
    
    if len(all_pairs) == 0:
        print("\nError: No pairs generated across all problems.")
        print("Try: --min-speedup 1.1 or --time-similarity 0.1")
        return
    
    # Overall statistics
    print(f"\n=== Overall Statistics ===")
    print(f"Total pairs across all problems: {len(all_pairs)}")
    all_speedups = [p['speedup_ratio'] for p in all_pairs]
    print(f"Global speedup range: {min(all_speedups):.2f}x - {max(all_speedups):.2f}x")
    print(f"Average speedup: {sum(all_speedups)/len(all_speedups):.2f}x")
    
    print(f"\nPairs per problem:")
    for problem_id, stats in problem_stats.items():
        print(f"  {problem_id}: {stats['pairs']} pairs from {stats['unique_solutions']} solutions")
    
    # Generate dataset entries
    print(f"\n=== Generating dataset entries ===")
    instructions = generate_instruction_variants()
    dataset = []
    
    for i, pair in enumerate(all_pairs):
        # Cycle through instruction variants
        instruction = instructions[i % len(instructions)]
        entry = generate_dataset_entry(pair, instruction, args.include_explanation)
        dataset.append(entry)
    
    # Write to JSONL
    output_path = Path(args.output)
    print(f"\nWriting {len(dataset)} entries to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Done! Dataset saved to {output_path}")
    print(f"\nDataset composition:")
    for inst in instructions:
        count = sum(1 for e in dataset if e['instruction'] == inst)
        print(f"  '{inst[:50]}...': {count} examples")
    
    print(f"\nExample entry:")
    example = dataset[0].copy()
    # Remove metadata for cleaner display
    if 'metadata' in example:
        del example['metadata']
    print(json.dumps(example, indent=2, ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    main()