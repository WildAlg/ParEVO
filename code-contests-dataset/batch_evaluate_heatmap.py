"""
Plot heatmaps of speedup results from merged CSV files.

Reads: batch_results_t{threads}.csv
Outputs: heatmap_t{threads}.pdf
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path


# Configuration - modify as needed
THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64]
CSV_DIR = Path(__file__).parent / "batch_results"
SEQUENTIAL_SETUP = "sequential"  # Excluded from heatmap

# Color configuration
COLORMAP = "Greens"  # Single-hue green, darker = higher speedup
INVALID_COLOR = "#D3D3D3"  # Light gray for ERR/empty cells
TEXT_COLOR_THRESHOLD = 0.6  # Use white text when cell is darker than this (normalized)
MAIN_FONT_SIZE = 16

def load_csv(csv_path: Path) -> tuple[list, list, dict]:
    """
    Load CSV and return (problem_ids, setups, data_dict).
    data_dict: {(problem_id, setup): value} where value is float or None
    """
    data = {}
    setups = []
    problem_ids = []
    
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return [], [], {}
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        setups = [col for col in reader.fieldnames if col != 'problem_id']
        
        for row in reader:
            problem_id = row.get('problem_id', '')
            if not problem_id:
                continue
            problem_ids.append(problem_id)
            
            for setup in setups:
                val = row[setup].strip()
                if val == '' or val == 'ERR':
                    data[(problem_id, setup)] = None
                else:
                    try:
                        data[(problem_id, setup)] = float(val)
                    except ValueError:
                        data[(problem_id, setup)] = None
    
    return problem_ids, setups, data


def plot_heatmap(thread_count: int):
    """
    Create and save a heatmap for the given thread count.
    """
    csv_path = CSV_DIR / f"batch_results_t{thread_count}.csv"
    problem_ids, setups, data = load_csv(csv_path)
    
    if not data:
        print(f"No data for t{thread_count}, skipping")
        return
    
    # Filter out sequential setup
    setups = [s for s in setups if s != SEQUENTIAL_SETUP]
    
    if not setups:
        print(f"No non-sequential setups for t{thread_count}, skipping")
        return
    
    # Build data matrix
    n_problems = len(problem_ids)
    n_setups = len(setups)
    
    matrix = np.full((n_problems, n_setups), np.nan)
    mask = np.zeros((n_problems, n_setups), dtype=bool)  # True for invalid cells
    
    for i, problem_id in enumerate(problem_ids):
        for j, setup in enumerate(setups):
            val = data.get((problem_id, setup))
            if val is not None:
                matrix[i, j] = val
            else:
                mask[i, j] = True
    
    # Determine color scale bounds
    valid_values = matrix[~np.isnan(matrix)]
    if len(valid_values) == 0:
        print(f"No valid data for t{thread_count}, skipping")
        return
    
    vmin = max(0, np.min(valid_values) * 0.9)
    vmax = np.max(valid_values) * 1.1
    
    # Create figure
    fig_width = max(8, n_setups * 1.2 + 2)
    fig_height = max(6, n_problems * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create colormap
    cmap = plt.cm.get_cmap(COLORMAP).copy()
    cmap.set_bad(INVALID_COLOR)
    
    # Plot heatmap
    masked_matrix = np.ma.array(matrix, mask=mask)
    im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Draw gray rectangles for invalid cells
    for i in range(n_problems):
        for j in range(n_setups):
            if mask[i, j]:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                  facecolor=INVALID_COLOR, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
    
    # Add text annotations
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(n_problems):
        for j in range(n_setups):
            if mask[i, j]:
                ax.text(j, i, '—', ha='center', va='center', 
                        color='#666666', fontsize=MAIN_FONT_SIZE, fontweight='bold')
            else:
                val = matrix[i, j]
                # Determine text color based on cell brightness
                normalized_val = norm(val)
                text_color = 'white' if normalized_val > TEXT_COLOR_THRESHOLD else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=text_color, fontsize=MAIN_FONT_SIZE)
    
    # Configure axes
    ax.set_xticks(np.arange(n_setups))
    ax.set_yticks(np.arange(n_problems))
    ax.set_xticklabels(setups, rotation=25, ha='right', fontsize=(MAIN_FONT_SIZE ))
    ax.set_yticklabels(problem_ids, fontsize=(MAIN_FONT_SIZE))
    
    # Labels and title
    # ax.set_xlabel('Setup', fontsize=12)
    # ax.set_ylabel('Problem', fontsize=12)
    ax.set_title(f'Speedup vs Sequential (threads={thread_count})', fontsize=(MAIN_FONT_SIZE + 2), fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Speedup (×)', fontsize=(MAIN_FONT_SIZE))
    
    # Add grid lines
    ax.set_xticks(np.arange(n_setups + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_problems + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = CSV_DIR / f"heatmap_t{thread_count}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    print(f"CSV Directory: {CSV_DIR}")
    print(f"Thread counts: {THREAD_COUNTS}")
    print(f"Colormap: {COLORMAP}")
    print()
    
    for thread_count in THREAD_COUNTS:
        plot_heatmap(thread_count)
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()