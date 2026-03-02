"""
Plots a histogram of runtimes for a specific benchmark configuration.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Input/Output
    parser.add_argument("input_csv", type=str, help="Input CSV file containing the results.")
    parser.add_argument("-o", "--output", type=str, default="runtime_hist.png", help="Output filename for the plot.")
    
    # Filters
    parser.add_argument("--name", type=str, required=True, help="Benchmark name (e.g., 'bw_decode/list_rank').")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads (e.g., 32).")
    parser.add_argument("--language", type=str, required=True, help="Language (e.g., 'rust', 'cpp').")
    parser.add_argument("--model", type=str, required=True, help="Parallelism model (e.g., 'rayon', 'omp', 'parlay').")
    
    # Plotting options
    parser.add_argument("--bins", type=int, default=30, help="Number of bins for the histogram.")
    
    return parser.parse_args()

def main():
    args = get_args()

    # 1. Load Data
    if not os.path.exists(args.input_csv):
        print(f" Error: File '{args.input_csv}' not found.")
        return

    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f" Error reading CSV: {e}")
        return

    # 2. Pre-process (Optional: Filter for valid runs only)
    # Most likely you only want to plot runtimes where is_valid == True
    if "is_valid" in df.columns:
        original_count = len(df)
        df = df[df["is_valid"] == True]
        if len(df) < original_count:
            print(f"  Filtered out {original_count - len(df)} invalid runs.")

    # 3. Apply Filters
    filtered_df = df[
        (df["name"] == args.name) &
        (df["num_threads"] == args.threads) &
        (df["language"] == args.language) &
        (df["parallelism_model"] == args.model)
    ]
    print(filtered_df)

    # 4. Check results
    n_samples = len(filtered_df)
    print(f"   Found {n_samples} samples matching configuration:")
    print(f"   Name: {args.name}")
    print(f"   Language: {args.language}")
    print(f"   Model: {args.model}")
    print(f"   Threads: {args.threads}")

    if filtered_df.empty:
        print(" No matching data found. Check your spelling or CSV content.")
        return

    # 5. Plotting
    runtimes = filtered_df["runtime"]
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Create Histogram with Kernel Density Estimate (KDE)
    sns.histplot(
        runtimes, 
        bins=args.bins, 
        kde=True, 
        color="skyblue", 
        edgecolor="black"
    )

    # Calculate stats for lines
    mean_val = runtimes.mean()
    median_val = runtimes.median()
    min_val = runtimes.min()

    # Add vertical lines for Mean and Median
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.5e}s')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.5e}s')

    # Formatting
    plt.title(f"Runtime Distribution: {args.name}\n({args.language} + {args.model} @ {args.threads} threads)", fontsize=14)
    plt.xlabel("Runtime (seconds)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    
    # Save output
    plt.tight_layout()
    plt.savefig(args.output)
    print(f" Plot saved to {args.output}")
    print(f" Stats -> Mean: {mean_val:.5e}s | Median: {median_val:.5e}s | Min: {min_val:.5e}s | Max: {runtimes.max():.5e}s")

if __name__ == "__main__":
    main()