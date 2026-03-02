import os
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def collect_results(log_dir, patterns, print_results=True):
    # Pattern to match a timing line
    timing_pattern = re.compile(
        r"(?P<graph>[\w\d]+) : .*?: '([\d.]+)', '([\d.]+)', '([\d.]+)', '([\d.]+)', '([\d.]+)', geomean = ([\d.]+)"
    )

    # Optional pattern to match the summary line at the bottom
    summary_pattern = re.compile(
        r".*? : (?P<proc>\d+) : geomean of mins = ([\d.]+), geomean of geomeans = ([\d.]+)"
    )

    results = defaultdict(dict)  # { filename: { graph_name: {times, geomean} } }

    for pattern in patterns:
        for file in Path(log_dir).glob(pattern):
            with open(file) as f:
                content = f.read()
                matches = timing_pattern.findall(content)
                for m in matches:
                    graph, *times, geomean = m
                    times = list(map(float, times))
                    geomean = float(geomean)
                    results[file.name][graph] = {
                        "times": times,
                        "geomean": geomean
                    }

                # Check for summary (optional)
                summary_match = summary_pattern.search(content)
                if summary_match:
                    proc = int(summary_match.group("proc"))
                    min_gmean = float(summary_match.group(2))
                    gmean_gmean = float(summary_match.group(3))
                    results[file.name]["_summary"] = {
                        "proc": proc,
                        "geomean_of_mins": min_gmean,
                        "geomean_of_geomeans": gmean_gmean
                    }

    # Example: print the parsed data nicely
    if print_results:
        for filename, graphs in sorted(results.items()):
            print(f"\n== {filename} ==")
            for graph, data in graphs.items():
                if graph == "_summary":
                    print(f"  Summary: proc={data['proc']}, geomean_of_mins={data['geomean_of_mins']}, geomean_of_geomeans={data['geomean_of_geomeans']}")
                else:
                    print(f"  {graph}:")
                    print(f"    Times: {data['times']}")
                    print(f"    Geomean: {data['geomean']}")
    return results

def collect_results_to_dataframe(results):
    # Prepare a list of rows for the DataFrame
    rows = []
    for filename, graphs in results.items():
        # # Extract thread count from filename, e.g., BFS_16.log -> 16
        # match = re.search(r'_(\d+)\.log$', filename)
        # if match is None:
        #     match = re.search(r'_(\d+)\_small.log$', filename)
        # thread = int(match.group(1)) if match else None

        # Detect thread count and whether it's a _small variant
        match = re.search(r'_(\d+)(?:_small)?\.log$', filename)
        thread = int(match.group(1)) if match else None
        is_small = "_small.log" in filename

        for graph, data in graphs.items():
            if graph == "_summary":
                row = {
                    "filename": filename,
                    "thread": thread,
                    "graph": "summary",
                    "geomean_of_mins": data["geomean_of_mins"],
                    "geomean_of_geomeans": data["geomean_of_geomeans"]
                }
            else:
                row = {
                    "filename": filename,
                    "thread": thread,
                    "graph": graph,
                    "times": data["times"],
                    "geomean": data["geomean"]
                }
                
            row["small"] = is_small
            rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(rows)
    # Add a 'type' column: 'Mix' if 'Mix' in filename, else 'Original'
    df['type'] = df['filename'].apply(lambda x: 'Mix' if 'Mix' in x else 'Original')
    return df

# Get unique graph types and line styles for Mix/Original
def plot_data(df_graphs, algorithm='', save_path=None):
    graph_types = [g for g in df_graphs['graph'].unique() if g != 'summary']
    linestyles = {'Original': 'solid', 'Mix': 'dashed'}
    markers = {'Original': 'o', 'Mix': 's'}
    colors = plt.cm.get_cmap('tab10', len(graph_types))

    plt.figure(figsize=(10, 6))

    for idx, graph_type in enumerate(graph_types):
        for t in ['Original', 'Mix']:
            subdf = df_graphs[(df_graphs['graph'] == graph_type) & (df_graphs['type'] == t)]
            if not subdf.empty:
                subdf = subdf.sort_values('thread')
                plt.plot(
                    subdf['thread'],
                    subdf['geomean'],
                    marker=markers[t],
                    linestyle=linestyles[t],
                    label=f"{graph_type} ({t})",
                    color=colors(idx)
                )

    plt.xlabel('Threads')
    plt.ylabel('Geomean (s)')
    plt.title(f'Geomean (s) vs Threads for Each Graph Type (Mix vs Original) for {algorithm}')
    plt.legend(title='Graph (Type)', 
                bbox_to_anchor=(1.05, 1),  # position legend just outside the right
                loc='upper left',
                borderaxespad=0.)
    plt.grid(True, which="both", ls="--")
    threads = sorted(df_graphs['thread'].unique())
    plt.xticks(threads, [str(t) for t in threads])
    # plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout(rect=[0, 0, 1.3, 1])  # room for legend
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_ratio_data(df_graphs, small=False, algorithm='', save_path=None):
    # Only keep non-summary rows
    df_graphs = df_graphs[df_graphs['graph'] != 'summary']

    # Get all unique graph names (e.g., road, kron, etc.)
    graph_types = df_graphs['graph'].unique()
    colors = plt.cm.get_cmap('tab10', len(graph_types))

    plt.figure(figsize=(10, 6))

    for idx, graph_type in enumerate(graph_types):
        if small:
            df_orig = df_graphs[(df_graphs['graph'] == graph_type) & 
                                (df_graphs['type'] == 'Original') & 
                                (df_graphs['small'])]

            df_mix = df_graphs[(df_graphs['graph'] == graph_type) & 
                            (df_graphs['type'] == 'Mix') & 
                            (df_graphs['small'])]
        else:
            df_orig = df_graphs[(df_graphs['graph'] == graph_type) & 
                                (df_graphs['type'] == 'Original') & 
                                (~df_graphs['small'])]

            df_mix = df_graphs[(df_graphs['graph'] == graph_type) & 
                            (df_graphs['type'] == 'Mix') & 
                            (~df_graphs['small'])]

        # Merge on thread count
        df_joined = pd.merge(df_orig, df_mix, on='thread', suffixes=('_orig', '_mix'))

        if df_joined.empty:
            continue

        # Compute the ratio: Original / Mix
        df_joined['ratio'] = df_joined['geomean_orig'] / df_joined['geomean_mix']
        df_joined = df_joined.sort_values('thread')

        plt.plot(
            df_joined['thread'],
            df_joined['ratio'],
            label=graph_type,
            marker='o',
            color=colors(idx)
        )

    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Threads')
    plt.ylabel('Geomean Ratio (Original / Mix)')
    plt.title(f'Ratio of Original to Mix Geomean vs Threads for {algorithm}')
    plt.grid(True, which="both", ls="--")
    threads = sorted(df_graphs['thread'].unique())
    plt.xticks(threads, [str(t) for t in threads])
    # plt.yscale('log')  # Optional: comment out if unnecessary
    plt.legend(title='Graph', 
               bbox_to_anchor=(1.05, 1),
               loc='upper left')
    plt.tight_layout(rect=[0, 0, 1.3, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# Obtain the implementation of serial BFS from PBBS
def read_file_strip_leading_comments(path):
    """
    Reads a file and strips the leading comment block at the top (// or /* ... */).
    Returns the rest of the file as a string.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    in_block_comment = False
    code_start_idx = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("//"):
            continue
        if stripped.startswith("/*"):
            in_block_comment = True
            continue
        if stripped == "":
            continue
        # First non-comment, non-empty line
        code_start_idx = idx
        break

    return "".join(lines[code_start_idx:])

def read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return ""
# Example usage:
# code = read_file_strip_leading_comments("pbbsbench/benchmarks/breadthFirstSearch/backForwardBFS/BFS.C")

def obtain_pbbs_implementation(algorithm, alg_dir):
    """
    Loads the main .C and .h files for a PBBS algorithm, as well as graph.h and graphIO.h.
    Returns: (file_c_stripped, file_h, graph_h, graph_io)
    """
    # Main C++ source file
    file_c_path = os.path.join(alg_dir, f"{algorithm}.C")
    file_h_path = os.path.join(alg_dir, f"{algorithm}.h")
    graph_h_path = os.path.join("pbbsbench", "common", "graph.h")
    graph_io_path = os.path.join("pbbsbench", "common", "graphIO.h")


    file_c = read_file_strip_leading_comments(file_c_path)
    file_h = read_file(file_h_path)
    graph_h = read_file_strip_leading_comments(graph_h_path)
    graph_io = read_file_strip_leading_comments(graph_io_path)

    # Optionally, strip comments or preprocess file_c if needed
    file_c_stripped = file_c  # or add your own stripping logic

    return file_c_stripped, file_h, graph_h, graph_io

# Modify filename to save the extracted code written by LLM in your intended directory
def extract_code(text, filename):
    # Regular expression to find C++ code blocks
    code_blocks = re.findall(r'```cpp\n(.*?)```', text, re.DOTALL)
    
    # If code blocks are found, return the first one (assuming there's only one)
    if code_blocks:
        cpp_code = code_blocks[0].strip()
        with open(filename, 'w') as file:
            file.write(cpp_code)
    else:
        return "No C++ code found."