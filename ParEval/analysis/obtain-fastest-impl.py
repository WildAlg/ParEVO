""" Compute the metrics over the data.
"""
# std imports
import argparse
import json
from math import comb
from typing import Union

# tpl imports
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=str, help="Input CSV file containing the test cases.")
    parser.add_argument("-k", "--k", type=int, nargs='+', default=[1,5,10,20], help="K value for pass@k, build@k, and speedup@k.")
    parser.add_argument("-n", "--n", type=int, default=1, help="N value for speedup@k.")
    parser.add_argument("-o", "--output", type=str, help="Output csv file containing the results.")
    parser.add_argument("--problem-sizes", type=str, default='../drivers/problem-sizes.json', help="Json with problem sizes. Used for calculating GPU efficiency.")
    return parser.parse_args()


def save_fastest_parlay_omp(df: pd.DataFrame, k: int, n: int, output_file: str):
    """
    Finds the fastest Parlay and OMP implementation for each problem
    and saves the corresponding code to a JSON file.
    """
    # The original speedupk function can still be used for data preparation.
    # It returns the filtered df and the ratio_df which we can use for finding fastest run.
    # It also has the filtering logic that we need, so let's reuse it.
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # choose processor count; hardcoded right now
    df = df[(df["parallelism_model"] == "serial") |
            (df["parallelism_model"] == "cuda") |
            (df["parallelism_model"] == "hip") |
            ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32))]
    df = df.copy()

    # final_df, avg_ratio = speedupk(df, k, n)

    # Filter the original DataFrame for just Parlay and OMP
    parlay_omp_df = df[df["parallelism_model"].isin(["parlay", "omp"])].copy()

    # Find the indices of the fastest run for each group.
    # We use .dropna() to prevent KeyErrors from empty groups.
    fastest_indices = parlay_omp_df.groupby(["name", "parallelism_model"])["runtime"].idxmin().dropna()

    # Use the indices to select the corresponding rows from the DataFrame.
    fastest_runs = parlay_omp_df.loc[fastest_indices].reset_index(drop=True)
    print("fastest_runs:", fastest_runs['generated_output'])


    # Select the columns to be saved
    output_data = fastest_runs[[
        "name",
        "problem_type",
        "parallelism_model",
        "runtime",
        "generated_output"
    ]].to_dict(orient="records")

    # Save as JSON
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Fastest implementations for Parlay and OMP saved to {output_file}")
  

def parse_problem_size(problem_size: str) -> int:
    """ problem size is of format '(1<<n)' """
    num = problem_size.split("<<")[1][:-1]
    return 2 ** int(num)

def main():
    args = get_args()

    # read in input
    df = pd.read_csv(args.input_csv)

    # read in problem sizes
    with open(args.problem_sizes, "r") as f:
        problem_sizes = json.load(f)
        for problem in problem_sizes:
            for parallelism_model, problem_size in problem_sizes[problem].items():
                df.loc[(df["name"] == problem) & (df["parallelism_model"] == parallelism_model), "problem_size"] = parse_problem_size(problem_size)

    # remove rows where parallelism_model is kokkos and num_threads is 64
    df = df[~((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 64))]

    # filter/aggregate
    df["did_run"] = df["did_run"].fillna(False)     # if it didn't build, then this will be nan; overwrite
    df["is_valid"] = df["is_valid"].fillna(False)   # if it didn't build, then this will be nan; overwrite

    # avg_ratios = []
    for k in args.k:
        save_fastest_parlay_omp(df, k, args.n, f"{args.output}/{k}.json")
        

if __name__ == "__main__":
    main()