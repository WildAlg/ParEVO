import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from itertools import cycle

def read_and_merge_csvs(file_paths):
    """Reads multiple CSV files and merges them into a single DataFrame."""
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        df["source"] = os.path.basename(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def extract_model_group(filename):
    """Extracts a base model group name from a filename for consistent coloring."""
    base = os.path.splitext(filename)[0]
    parts = base.split("-")
    if "omp" in parts[-1].lower() or "parlay" in parts[-1].lower():
        return "-".join(parts[:-1])
    return base

def plot_ratio_histogram(df_to_plot, output_file, title, group_to_color, log_scale=False):
    """
    Generates and saves a bar chart of Parlay/OMP ratios for a given set of metrics.
    
    Args:
        df_to_plot (pd.DataFrame): DataFrame containing the data to plot.
        output_file (str): Path to save the output plot image.
        title (str): The title for the plot.
        group_to_color (dict): A dictionary mapping model groups to colors.
        log_scale (bool): Whether to use a logarithmic scale for the y-axis.
    """
    # Sort metrics for a consistent ordering on the x-axis
    metrics = sorted(df_to_plot["Metric"].unique())
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(25, 9))
    
    sources = sorted(df_to_plot["source"].unique())
    n_sources = len(sources)
    width = 0.8 / n_sources

    # Use a copy of the group_to_color keys for consistent hatching logic
    group_counts = {group: 0 for group in group_to_color.keys()}
    
    handles = []

    for i, source in enumerate(sources):
        group = extract_model_group(source)
        base_color = group_to_color.get(group, 'gray')

        # Use hatching for the second file of the same model group to differentiate them
        hatch = "//" if group_counts[group] > 0 else ""
        group_counts[group] += 1
        
        # Prepare data for this source, ensuring it aligns with the plot's metrics
        sub_df = df_to_plot[df_to_plot["source"] == source]
        source_ratios = sub_df.set_index('Metric')['ratio']
        values = source_ratios.reindex(metrics).values

        rects = ax.bar(x + (i - n_sources/2) * width,
                       values, width,
                       label=source,
                       color=base_color,
                       hatch=hatch,
                       edgecolor="black")
        handles.append(rects)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Parlay / OMP Ratio" + (" (log scale)" if log_scale else ""))
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    
    ax.legend(handles, sources, title="Files", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Adjust layout to prevent the legend from being cut off
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_file, dpi=300)
    plt.close(fig) # Close the figure to free up memory
    print(f"Plot saved to {output_file}")

def plot_values_histogram(file_path, output_file="metrics_values_histogram.png", log_scale=False):
    """Generates a bar chart comparing absolute Parlay and OMP values."""
    df = pd.read_csv(file_path)
    df = df[df["OMP"].notna()]
    metrics = df["Metric"].values
    x = np.arange(len(metrics))
    parlay_values = df["Parlay"].values
    omp_values = df["OMP"].values

    width = 0.35
    fig, ax = plt.subplots(figsize=(25, 9))
    rects1 = ax.bar(x - width/2, parlay_values, width, label="Parlay", color='tab:blue', edgecolor='black')
    rects2 = ax.bar(x + width/2, omp_values, width, label="OMP", color='tab:orange', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Value" + (" (log scale)" if log_scale else ""))
    ax.set_title(f"Parlay vs OMP Values: {os.path.basename(file_path)}")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CSV metrics")
    parser.add_argument("--files", nargs="+", required=True, help="Input CSV files")
    parser.add_argument("-o", "--output", default="histogram.png", help="Base output plot filename. Will be suffixed with '_speedup' and '_efficiency'.")
    parser.add_argument("--log-scale", action="store_true", help="Use logarithmic scale for y-axis")
    parser.add_argument("--plot-values", action="store_true",
                        help="Plot histogram of Parlay and OMP values for a single CSV file instead of ratios")
    args = parser.parse_args()

    if args.plot_values:
        if len(args.files) != 1:
            raise ValueError("--plot-values mode only supports a single CSV file")
        plot_values_histogram(args.files[0], args.output, args.log_scale)
    else:
        # --- Data Preparation ---
        df = read_and_merge_csvs(args.files)
        df = df[df["OMP"].notna()]
        df["ratio"] = df["Parlay"] / df["OMP"]
        df["model_group"] = df["source"].apply(extract_model_group)

        # --- Consistent Color Mapping ---
        all_groups = sorted(df["model_group"].unique())
        base_colors = plt.cm.tab10.colors
        color_cycle = cycle(base_colors)
        group_to_color = {group: next(color_cycle) for group in all_groups}
        
        # --- Filter Data for Each Plot ---
        speedup_prefixes = ('pass@1', 'speedup@', 'speedup_max@')
        efficiency_prefixes = ('efficiency@', 'efficiency_max@')

        df_speedup = df[df['Metric'].str.startswith(speedup_prefixes)].copy()
        df_efficiency = df[df['Metric'].str.startswith(efficiency_prefixes)].copy()

        # --- Generate Plots ---
        base_output, ext = os.path.splitext(args.output)
        
        # Plot 1: Speedup Metrics
        if not df_speedup.empty:
            output_speedup = f"{base_output}_speedup{ext}"
            plot_ratio_histogram(
                df_to_plot=df_speedup,
                output_file=output_speedup,
                title="Parlay vs OMP Ratio: Speedup Metrics",
                group_to_color=group_to_color,
                log_scale=args.log_scale
            )
        else:
            print("No data found for speedup metrics. Skipping that plot.")
            
        # Plot 2: Efficiency Metrics
        if not df_efficiency.empty:
            output_efficiency = f"{base_output}_efficiency{ext}"
            plot_ratio_histogram(
                df_to_plot=df_efficiency,
                output_file=output_efficiency,
                title="Parlay vs OMP Ratio: Efficiency Metrics",
                group_to_color=group_to_color,
                log_scale=args.log_scale
            )
        else:
            print("No data found for efficiency metrics. Skipping that plot.")
