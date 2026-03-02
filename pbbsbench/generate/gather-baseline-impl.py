import json
import os
import glob
from pathlib import Path

# --- Configuration ---
# Root directory where 'benchmarks' folder is located
ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT_DIR / "benchmarks"

# Input/Output files
INPUT_JSON = ROOT_DIR / "prompts" / "prompts-with-description.json"
OUTPUT_JSON = ROOT_DIR / "generate" / "outputs" / "prompts-with-baseline-impl.json"

# Mapping from directory name to C file prefix/name
BENCHMARK_MAP = {
    "integerSort/parallelRadixSort": "isort",
    "integerSort/serialRadixSort": "isort",
    "comparisonSort/sampleSort": "sort",
    "comparisonSort/quickSort": "sort",
    "comparisonSort/mergeSort": "sort",
    "comparisonSort/stableSampleSort": "sort",
    "comparisonSort/serialSort": "sort",
    "comparisonSort/ips4o": "sort",
    "removeDuplicates/serial_hash": "dedup",
    "removeDuplicates/serial_sort": "dedup",
    "removeDuplicates/parlayhash": "dedup",
    "histogram/sequential": "histogram",
    "histogram/parallel": "histogram", # Handle special case if ext is included
    "wordCounts/histogram": "wc",
    "wordCounts/serial": "wc",
    "invertedIndex/sequential": "index",
    "invertedIndex/parallel": "index",
    "suffixArray/parallelKS": "SA",
    "suffixArray/parallelRange": "SA",
    "suffixArray/serialDivsufsort": "SA",
    "longestRepeatedSubstring/doubling": "lrs",
    "classify/decisionTree": "classify",
    "minSpanningForest/parallelFilterKruskal": "MST",
    "minSpanningForest/serialMST": "MST",
    "spanningForest/incrementalST": "ST",
    "spanningForest/ndST": "ST",
    "spanningForest/serialST": "ST",
    "breadthFirstSearch/simpleBFS": "BFS",
    "breadthFirstSearch/backForwardBFS": "BFS",
    "breadthFirstSearch/deterministicBFS": "BFS",
    "breadthFirstSearch/serialBFS": "BFS",
    "maximalMatching/serialMatching": "matching",
    "maximalMatching/incrementalMatching": "matching",
    "maximalIndependentSet/incrementalMIS": "MIS",
    "maximalIndependentSet/ndMIS": "MIS",
    "maximalIndependentSet/serialMIS": "MIS",
    "nearestNeighbors/octTree": "neighbors",
    "rayCast/kdTree": "ray",
    "convexHull/quickHull": "hull",
    "convexHull/serialHull": "hull",
    "delaunayTriangulation/incrementalDelaunay": "delaunay",
    "delaunayRefine/incrementalRefine": "refine",
    "rangeQuery2d/parallelPlaneSweep": "range",
    "rangeQuery2d/serial": "range",
    "nBody/parallelCK": "nbody"
}

def find_and_read_code(bench_name):
    """
    Locates the .C file for a given benchmark name.
    """
    if bench_name not in BENCHMARK_MAP:
        print(f"⚠️  Warning: '{bench_name}' not found in map.")
        return None, None

    file_prefix = BENCHMARK_MAP[bench_name]
    
    # Construct directory path
    dir_path = BENCHMARK_DIR / bench_name
    
    if not dir_path.exists():
        print(f"❌ Error: Directory not found: {dir_path}")
        return None, file_prefix

    # Determine exact filename
    target_file = dir_path / f"{file_prefix}.C"
    # Try to find the file
    if not target_file.exists():
        target_file = dir_path / f"{file_prefix}.h"
        

    # Read content
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            return f.read(), file_prefix
    except Exception as e:
        print(f"❌ Error reading {target_file}: {e}")
        return None, file_prefix

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Input file {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    processed_count = 0
    
    for entry in data:
        name = entry.get("name")
        
        if not name:
            continue

        print(f"Processing: {name}...")
        code_content, alg_name = find_and_read_code(name)
        entry["alg_name"] = alg_name
        entry["generate_model"] = "baseline"
        if code_content:
            # Create the list format as requested
            # We wrap it in a dictionary structure compatible with your previous script
            entry["outputs"] = [
                code_content
            ]
            processed_count += 1
        else:
            # Initialize empty if not found so script doesn't break
            entry["outputs"] = []

    # Save result
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"\n✅ Done. Processed {processed_count} benchmarks.")
    print(f"Saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()