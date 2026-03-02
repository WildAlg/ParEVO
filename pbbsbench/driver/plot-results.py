#!/usr/bin/env python3
"""
plot-results.py

Plots benchmark results from the processed JSON produced by process-logfile.py.

What it plots (no seaborn, no explicit colors):
  1) overall geomean vs threads (one line per (model,mix,algorithm,scheduler,server,input_size))
  2) per-graph geomean vs threads (one subplot per graph name)

Usage:
  python plot-results.py -i processed_benchmark_data.json -o plots/
  python plot-results.py -i processed_benchmark_data.json --filter "algorithm=backForwardBFS,scheduler=omp"
"""

import os
import json
import math
import argparse
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def parse_filter_kv(s: Optional[str]) -> Dict[str, str]:
    """
    Parse filter string like:
      "algorithm=backForwardBFS,scheduler=omp,model=gemini-2.5-pro"
    """
    if not s:
        return {}
    out: Dict[str, str] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad filter token '{part}', expected key=value")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def matches_filter(entry: Dict[str, Any], flt: Dict[str, str]) -> bool:
    for k, v in flt.items():
        if k not in entry:
            return False
        # compare as strings to handle mix_value ints vs strings
        if str(entry[k]) != v:
            return False
    return True


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def label_for_entry(e: Dict[str, Any]) -> str:
    # Keep label readable but unique-ish
    return (
        f"{e.get('model')} | mix={e.get('mix_value')} | {e.get('algorithm')} | "
        f"{e.get('scheduler')} | {e.get('server')} | {e.get('input_size')}"
    )


def filename_stub_for_entry(e: Dict[str, Any]) -> str:
    # filesystem-safe stub
    parts = [
        str(e.get("model", "unknown")),
        f"mix{e.get('mix_value', 'na')}",
        str(e.get("algorithm", "alg")),
        str(e.get("scheduler", "sched")),
        str(e.get("server", "server")),
        str(e.get("input_size", "input")),
    ]
    stub = "_".join(parts)
    stub = stub.replace("/", "-")
    return stub


def extract_thread_series(entry: Dict[str, Any]) -> Tuple[List[int], List[Optional[float]]]:
    """
    Returns sorted threads and overall_geomean values (None if missing/error).
    """
    rbt = entry.get("results_by_threads", {})
    threads: List[int] = []
    values: List[Optional[float]] = []
    for k, v in rbt.items():
        t = v.get("threads")
        if t is None:
            try:
                t = int(k)
            except Exception:
                continue
        og = safe_float(v.get("overall_geomean"))
        # If there's an error and no value, keep None
        if og is None and v.get("error"):
            og = None
        threads.append(int(t))
        values.append(og)

    # sort by threads
    order = sorted(range(len(threads)), key=lambda i: threads[i])
    threads = [threads[i] for i in order]
    values = [values[i] for i in order]
    return threads, values


def plot_overall_geomean(entries: List[Dict[str, Any]], out_dir: str, title_prefix: str = "") -> None:
    """
    One figure with multiple lines: overall geomean vs threads.
    """
    if not entries:
        return

    plt.figure()
    for e in entries:
        threads, vals = extract_thread_series(e)
        xs = []
        ys = []
        for t, v in zip(threads, vals):
            if v is None:
                continue
            xs.append(t)
            ys.append(v)
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", label=label_for_entry(e))

    plt.xscale("log", base=2)  # threads are usually powers of 2-ish
    plt.yscale("log")          # runtimes/geomeans often span orders of magnitude
    plt.xlabel("Threads")
    plt.ylabel("Overall geomean runtime (s)")
    plt.title(f"{title_prefix}Overall geomean vs threads".strip())
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize="small")

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "overall_geomean_vs_threads.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote {out_path}")


def collect_graph_names(entry: Dict[str, Any]) -> List[str]:
    graphs = set()
    for _, tdata in entry.get("results_by_threads", {}).items():
        ig = tdata.get("individual_geomeans", {}) or {}
        graphs.update(ig.keys())
    return sorted(graphs)


def plot_per_graph(entry: Dict[str, Any], out_dir: str) -> None:
    """
    For a single entry, make a figure with one subplot per graph:
      graph geomean vs threads
    """
    graphs = collect_graph_names(entry)
    if not graphs:
        return

    # Determine layout
    n = len(graphs)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(7 * cols, 3.5 * rows))
    fig.suptitle(label_for_entry(entry), fontsize=12)

    # Precompute sorted thread list
    threads_sorted = sorted(
        {int(v.get("threads", int(k))) for k, v in entry.get("results_by_threads", {}).items()}
    )

    for i, g in enumerate(graphs, start=1):
        ax = fig.add_subplot(rows, cols, i)

        xs = []
        ys = []
        for t in threads_sorted:
            # Find thread record
            tdata = entry["results_by_threads"].get(str(t))
            if tdata is None:
                # sometimes key is not str(thread); try scan
                for kk, vv in entry["results_by_threads"].items():
                    if int(vv.get("threads", -1)) == t:
                        tdata = vv
                        break
            if not tdata:
                continue
            v = safe_float((tdata.get("individual_geomeans") or {}).get(g))
            if v is None:
                continue
            xs.append(t)
            ys.append(v)

        if xs:
            ax.plot(xs, ys, marker="o")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
        ax.set_title(g)
        ax.set_xlabel("Threads")
        ax.set_ylabel("Geomean (s)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    ensure_dir(out_dir)
    stub = filename_stub_for_entry(entry)
    out_path = os.path.join(out_dir, f"{stub}_per_graph.png")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to processed JSON file")
    parser.add_argument("-o", "--out_dir", default="plots", help="Output directory for plots")
    parser.add_argument(
        "--filter",
        default=None,
        help="Comma-separated key=value filters on top-level entry fields, e.g. "
             "'algorithm=backForwardBFS,scheduler=omp,server=XXXXXX'",
    )
    parser.add_argument(
        "--per_graph",
        action="store_true",
        help="Also generate per-entry per-graph subplot figures",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    flt = parse_filter_kv(args.filter)
    selected = [e for e in data if matches_filter(e, flt)]

    if not selected:
        print("No entries matched the filter." if flt else "No entries found.")
        return

    ensure_dir(args.out_dir)

    # 1) Combined plot across selected entries
    title_prefix = ""
    if flt:
        title_prefix = " | ".join([f"{k}={v}" for k, v in flt.items()]) + " — "
    plot_overall_geomean(selected, args.out_dir, title_prefix=title_prefix)

    # 2) Optional: per-graph plots per entry
    if args.per_graph:
        for e in selected:
            plot_per_graph(e, args.out_dir)


if __name__ == "__main__":
    main()
