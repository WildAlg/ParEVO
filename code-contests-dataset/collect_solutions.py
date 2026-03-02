#!/usr/bin/env python3
"""
Simple script to copy best solutions from result directories to a consolidated solutions folder.

Usage:
    python collect_solutions.py
"""

import shutil
from pathlib import Path

# Hardcoded configuration
PROBLEM_IDS = [
    'ccc15s1',
    'ccc17s3',
    'cco08p4',
    'coci11c1p3',
    'coci19c1p3',
    'coci21c2p1',
    'coci23c2p2',
    'joi21op1',
]

# Map: setup_id -> (source_dir, extension)
SETUPS = {
    "gemini-2.5-pro-cpp": ("results_gemini-2.5-pro_cpp", "cpp"),
    "gemini-2.5-pro-parlay-cpp": ("results_gemini-2.5-pro-finetuned_cpp", "cpp"),
    # Add more setups here
    # "some-rust-setup": ("results_rust_setup", "rs"),
}

OUTPUT_DIR = Path("./solutions")


def main():
    for setup_id, (source_dir, ext) in SETUPS.items():
        source_path = Path(source_dir)
        dest_dir = OUTPUT_DIR / setup_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {setup_id}...")
        
        for problem_id in PROBLEM_IDS:
            src_file = source_path / problem_id / f"best_solution.{ext}"
            dst_file = dest_dir / f"{problem_id}.{ext}"
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"  Copied {problem_id}")
            else:
                print(f"  Missing {problem_id}")


if __name__ == "__main__":
    main()