#!/usr/bin/env python3
"""
compare-runtime.py

Compares Gemini-generated runtime JSON vs baseline runtime JSON.

Inputs:
  - baseline_runtime.json  (model=baseline, mix_value=baseline)
  - processed-gemini-2.5-pro.json (model=gemini-2.5-pro, mix_value=int, etc.)

Matching rule (baseline selection):
  Match on (algorithm, scheduler, server, input_size).

Per-thread comparison:
  - overall speedup: baseline_overall / gemini_overall
  - per-graph speedups on intersection of graphs

Outputs:
  - comparison_summary.csv
  - comparison_per_graph.csv
  - plots:
      * overall_geomean_vs_threads__<...>.png
      * speedup_vs_threads__<...>.png
      * per_graph_speedup__<...>__t<threads>.png  (optional top-K)

Usage:
  python compare-runtime.py \
    --baseline baseline_runtime.json \
    --gemini processed-gemini-2.5-pro.json \
    --out-dir compare_out

Optional:
  --filter-algorithm backForwardBFS
  --filter-scheduler omp
  --filter-server XXXXXX
  --filter-input-size large
  --topk-graphs 20
"""

import os
import json
import math
import csv
import argparse
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


Key = Tuple[str, str, str, str]  # (algorithm, scheduler, server, input_size)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def geomean(xs: List[float]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None and x > 0]
    if not xs2:
        return None
    s = sum(math.log(x) for x in xs2)
    return math.exp(s / len(xs2))


def norm_threads_key(k: Any) -> Optional[int]:
    try:
        return int(k)
    except Exception:
        return None


def build_index(items: List[Dict[str, Any]]) -> Dict[Key, Dict[str, Any]]:
    """
    Build an index for fast lookup by (algorithm, scheduler, server, input_size).
    If duplicates exist, last one wins.
    """
    idx: Dict[Key, Dict[str, Any]] = {}
    for it in items:
        alg = it.get("algorithm")
        sch = it.get("scheduler")
        srv = it.get("server")
        insz = it.get("input_size")
        if not all([alg, sch, srv, insz]):
            continue
        idx[(alg, sch, srv, insz)] = it
    return idx


def extract_thread_block(entry: Dict[str, Any], t: int) -> Optional[Dict[str, Any]]:
    rbt = entry.get("results_by_threads", {})
    # keys may be ints or strings; try both
    if str(t) in rbt:
        return rbt[str(t)]
    if t in rbt:
        return rbt[t]
    # fallback: scan keys
    for k, v in rbt.items():
        kk = norm_threads_key(k)
        if kk == t:
            return v
    return None


def graphs_intersection(base_blk: Dict[str, Any], gem_blk: Dict[str, Any]) -> List[str]:
    bg = set((base_blk.get("individual_geomeans") or {}).keys())
    gg = set((gem_blk.get("individual_geomeans") or {}).keys())
    return sorted(bg.intersection(gg))


def compute_overall_from_intersection(block: Dict[str, Any], graphs: List[str]) -> Optional[float]:
    ig = block.get("individual_geomeans") or {}
    vals = []
    for g in graphs:
        v = ig.get(g)
        if isinstance(v, (int, float)) and v > 0:
            vals.append(float(v))
    return geomean(vals)


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except Exception:
        return None


def should_skip_block(block: Optional[Dict[str, Any]]) -> bool:
    if block is None:
        return True
    # Treat abnormal termination or explicit error as "invalid for comparison"
    if block.get("abnormal_terminated") is True:
        return True
    # NOTE: baseline may store command string in "error" even on success in your snippet.
    # So we only skip if error is a non-empty string AND overall_geomean is None AND no individual_geomeans.
    err = block.get("error")
    overall = block.get("overall_geomean")
    ig = block.get("individual_geomeans") or {}
    if err and (overall is None) and (len(ig) == 0):
        return True
    return False


def plot_overall_curves(out_path: str, threads: List[int], base_vals: List[float], gem_vals: List[float], title: str):
    plt.figure()
    plt.plot(threads, base_vals, marker="o", label="baseline (overall geomean)")
    plt.plot(threads, gem_vals, marker="o", label="gemini (overall geomean)")
    plt.xlabel("threads")
    plt.ylabel("time (lower is better)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_speedup_curve(out_path: str, threads: List[int], speedups: List[float], title: str):
    plt.figure()
    plt.plot(threads, speedups, marker="o")
    plt.axhline(1.0)
    plt.xlabel("threads")
    plt.ylabel("speedup = baseline / gemini (higher is better)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_graph_speedup(out_path: str, graph_names: List[str], speedups: List[float], title: str):
    # Sort by speedup descending, keep in sync
    pairs = sorted(zip(graph_names, speedups), key=lambda x: x[1], reverse=True)
    g2 = [p[0] for p in pairs]
    s2 = [p[1] for p in pairs]

    plt.figure(figsize=(10, max(3, 0.35 * len(g2))))
    plt.barh(range(len(g2)), s2)
    plt.yticks(range(len(g2)), g2)
    plt.gca().invert_yaxis()
    plt.axvline(1.0)
    plt.xlabel("speedup = baseline / gemini")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline_runtime.json")
    ap.add_argument("--gemini", required=True, help="processed-gemini-2.5-pro.json")
    ap.add_argument("--out-dir", default="compare_out")
    ap.add_argument("--filter-algorithm", default=None)
    ap.add_argument("--filter-scheduler", default=None)
    ap.add_argument("--filter-server", default=None)
    ap.add_argument("--filter-input-size", default=None)
    ap.add_argument("--topk-graphs", type=int, default=20, help="Per-graph plot shows up to top-K graphs.")
    ap.add_argument("--plot-per-graph", action="store_true", help="Emit per-graph bar charts for each thread.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    baseline_items = load_json(args.baseline)
    gemini_items = load_json(args.gemini)

    if not isinstance(baseline_items, list):
        raise ValueError("baseline_runtime.json must be a list of objects")
    if not isinstance(gemini_items, list):
        raise ValueError("processed gemini json must be a list of objects")

    baseline_idx = build_index(baseline_items)

    summary_csv = os.path.join(args.out_dir, "comparison_summary.csv")
    per_graph_csv = os.path.join(args.out_dir, "comparison_per_graph.csv")

    # Write CSV headers
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "mix_value", "algorithm", "scheduler", "server", "input_size", "threads",
            "baseline_overall", "gemini_overall", "speedup_baseline_over_gemini",
            "baseline_overall_from_intersection", "gemini_overall_from_intersection", "speedup_from_intersection",
            "n_graphs_intersection",
        ])

    with open(per_graph_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "mix_value", "algorithm", "scheduler", "server", "input_size", "threads",
            "graph", "baseline_geomean", "gemini_geomean", "speedup_baseline_over_gemini",
        ])

    # Iterate gemini entries, match baseline
    for gem in gemini_items:
        model = gem.get("model")
        mix_value = gem.get("mix_value")
        algorithm = gem.get("algorithm")
        scheduler = gem.get("scheduler")
        server = gem.get("server")
        input_size = gem.get("input_size")

        if args.filter_algorithm and algorithm != args.filter_algorithm:
            continue
        if args.filter_scheduler and scheduler != args.filter_scheduler:
            continue
        if args.filter_server and server != args.filter_server:
            continue
        if args.filter_input_size and input_size != args.filter_input_size:
            continue

        key: Key = (algorithm, scheduler, server, input_size)
        base = baseline_idx.get(key)
        if base is None:
            print(f"[WARN] No baseline match for {key}. Skipping.")
            continue

        # Determine common threads
        gem_threads = []
        for k in (gem.get("results_by_threads") or {}).keys():
            t = norm_threads_key(k)
            if t is not None:
                gem_threads.append(t)
        base_threads = []
        for k in (base.get("results_by_threads") or {}).keys():
            t = norm_threads_key(k)
            if t is not None:
                base_threads.append(t)

        common_threads = sorted(set(gem_threads).intersection(set(base_threads)))
        if not common_threads:
            print(f"[WARN] No common threads for {key}. Skipping.")
            continue

        # For plotting overall curves for this gemini entry
        plot_threads: List[int] = []
        plot_base_overall: List[float] = []
        plot_gem_overall: List[float] = []
        plot_speedups: List[float] = []

        for t in common_threads:
            gb = extract_thread_block(gem, t)
            bb = extract_thread_block(base, t)

            if should_skip_block(gb) or should_skip_block(bb):
                continue

            gem_overall = safe_float(gb.get("overall_geomean"))
            base_overall = safe_float(bb.get("overall_geomean"))

            # Compare on intersection of graphs to avoid “mixed dataset” issues
            inter = graphs_intersection(bb, gb)
            base_inter_overall = compute_overall_from_intersection(bb, inter) if inter else None
            gem_inter_overall = compute_overall_from_intersection(gb, inter) if inter else None

            speedup = (base_overall / gem_overall) if (base_overall and gem_overall and gem_overall > 0) else None
            speedup_inter = (base_inter_overall / gem_inter_overall) if (
                base_inter_overall and gem_inter_overall and gem_inter_overall > 0
            ) else None

            # Write summary row
            with open(summary_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    model, mix_value, algorithm, scheduler, server, input_size, t,
                    base_overall, gem_overall, speedup,
                    base_inter_overall, gem_inter_overall, speedup_inter,
                    len(inter),
                ])

            # Per-graph rows (intersection only)
            ig_b = bb.get("individual_geomeans") or {}
            ig_g = gb.get("individual_geomeans") or {}

            for gname in inter:
                bval = safe_float(ig_b.get(gname))
                gval = safe_float(ig_g.get(gname))
                if bval is None or gval is None or gval <= 0:
                    continue
                sp = bval / gval
                with open(per_graph_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        model, mix_value, algorithm, scheduler, server, input_size, t,
                        gname, bval, gval, sp
                    ])

            # Add to plot series (prefer intersection-based overall if available)
            if base_inter_overall and gem_inter_overall and gem_inter_overall > 0:
                plot_threads.append(t)
                plot_base_overall.append(base_inter_overall)
                plot_gem_overall.append(gem_inter_overall)
                plot_speedups.append(base_inter_overall / gem_inter_overall)

            # Optional per-graph bar plot for this thread
            if args.plot_per_graph and inter:
                # pick top-k by absolute baseline time (or you can sort by speedup)
                # Here: plot top-k by |baseline| (descending) to focus on heavy cases.
                weighted = []
                for gname in inter:
                    bval = safe_float(ig_b.get(gname))
                    gval = safe_float(ig_g.get(gname))
                    if bval is None or gval is None or gval <= 0:
                        continue
                    weighted.append((gname, bval, bval / gval))
                weighted.sort(key=lambda x: x[1], reverse=True)
                top = weighted[: max(1, args.topk_graphs)]
                gnames = [x[0] for x in top]
                sps = [x[2] for x in top]

                safe_tag = f"{model}_{mix_value}_{algorithm}_{scheduler}_{server}_{input_size}".replace("/", "_")
                outp = os.path.join(args.out_dir, f"per_graph_speedup__{safe_tag}__t{t}.png")
                plot_per_graph_speedup(
                    outp, gnames, sps,
                    title=f"Per-graph speedup (baseline/gemini)\n{safe_tag} | threads={t}"
                )

        # Plot overall curves for this entry if we collected points
        if len(plot_threads) >= 2:
            safe_tag = f"{model}_{mix_value}_{algorithm}_{scheduler}_{server}_{input_size}".replace("/", "_")
            out1 = os.path.join(args.out_dir, f"overall_geomean_vs_threads__{safe_tag}.png")
            out2 = os.path.join(args.out_dir, f"speedup_vs_threads__{safe_tag}.png")

            plot_overall_curves(
                out1, plot_threads, plot_base_overall, plot_gem_overall,
                title=f"Overall (intersection-based) geomean vs threads\n{safe_tag}"
            )
            plot_speedup_curve(
                out2, plot_threads, plot_speedups,
                title=f"Speedup vs threads (baseline/gemini)\n{safe_tag}"
            )

    print(f"Done.\nWrote:\n  {summary_csv}\n  {per_graph_csv}\nPlots in: {args.out_dir}")


if __name__ == "__main__":
    main()
