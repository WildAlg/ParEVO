import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_runtime_ratio(json_file_path: str, output_path):
    """
    Reads a JSON file containing runtime data for Parlay and OMP,
    calculates the ratio of Parlay runtime to OMP runtime, and
    plots a bar chart of the ratios for each problem.
    
    Args:
        json_file_path (str): The path to the JSON file.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON.")
        return

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    # Pivot the DataFrame to prepare for plotting
    pivot_df = df.pivot(index="name", columns="parallelism_model", values="runtime")

    # Calculate the ratio of Parlay to OMP runtimes
    # Use .get() to handle cases where 'parlay' or 'omp' might be missing
    pivot_df['parlay_omp_ratio'] = pivot_df.get('parlay', pd.Series()) / pivot_df.get('omp', pd.Series())

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot a single bar for each problem, representing the ratio
    problem_names = pivot_df.index
    x = np.arange(len(problem_names))
    ratios = pivot_df['parlay_omp_ratio']

    ax.bar(x, ratios)

    # Configure the plot labels and title
    ax.set_xlabel("Problem Name", fontweight='bold')
    ax.set_ylabel("Parlay / OMP Runtime Ratio", fontweight='bold')
    ax.set_title("Runtime Ratio: Parlay vs. OMP")
    ax.set_xticks(x)
    # ax.set_xticklabels(problem_names, rotation=45, ha='right')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Ratio = 1.0 (Equal Runtime)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()

    # Save the plot to a file
    fig.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

    # Close the figure to free memory
    plt.close(fig)


# Example usage with a dummy file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and plot performance data from a CSV file.')
    parser.add_argument('-i', '--input_file', 
                        type=str, 
                        required=True,
                        help='Path to the input CSV data file.')
    parser.add_argument('-o', '--output_path',
                        type=str,
                        help='Path to save the generated plot. Default is "plots".')
    
    args = parser.parse_args()

    plot_runtime_ratio("fastest_parlay_omp_k1.json")
