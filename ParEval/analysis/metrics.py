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
    parser.add_argument("--model-name", type=str, help="Add model name column with this value")
    parser.add_argument("--problem-wise", action="store_true", help="If turned on, generate problem-wise metrics in addition to problem-type-wise metrics.")
    return parser.parse_args()

def get_correctness_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Group by name, parallelism_model, and output_idx, and set is_valid to true only if all rows in the group have is_valid = true.
        Set it to false otherwise.
    """
    # group all the runs for this LLM output
    df = df.copy()
    agg = df.groupby(["name", "parallelism_model", "output_idx"]).agg({"is_valid": ["count", "sum"]})
    agg.columns = ["count", "sum"]

    # mark as valid only if all runs are valid
    agg["is_valid"] = agg["count"] == agg["sum"]
    agg = agg.reset_index()
    agg = agg.drop(columns=["count", "sum"])
    
    # add problem_type column from df
    agg = agg.merge(df[["name", "problem_type"]].drop_duplicates(), on="name", how="left")

    return agg

def nCr(n: int, r: int) -> int:
    if n < r:
        return 1
    return comb(n, r)

def buildk(df: pd.DataFrame, k: int, problem_wise: bool = False) -> pd.DataFrame:
    """ Compute the build@k metric """
    agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg({"did_build": ["count", "sum"]})
    agg.columns = ["total_build_attempts", "successful_builds"]
    agg = agg.reset_index()
    agg[f"build@{k}"] = agg.apply(lambda x: _passk(x["total_build_attempts"], x["successful_builds"], k), axis=1)
    
    if problem_wise:
        return agg.groupby(["parallelism_model", "name"]).agg({f"build@{k}": "mean"})
    else:
        return agg.groupby(["parallelism_model", "problem_type"]).agg({f"build@{k}": "mean"})

def _passk(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))

def passk(df: pd.DataFrame, k: int, problem_wise: bool = False) -> pd.DataFrame:
    """ Compute the pass@k metric """
    agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg({"is_valid": ["count", "sum"]})
    agg.columns = ["total_runs", "valid_count"]
    agg = agg.reset_index()
    agg[f"pass@{k}"] = agg.apply(lambda x: _passk(x["total_runs"], x["valid_count"], k), axis=1)
    
    if problem_wise:
        return agg.groupby(["parallelism_model", "name"]).agg({f"pass@{k}": "mean"})
    else:
        return agg.groupby(["parallelism_model", "problem_type"]).agg({f"pass@{k}": "mean"})

def _speedupk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int, col_name: str = 'speedup@{}') -> float:
    """ Compute the speedup@k metric """
    # create a copy of the runtimes
    if isinstance(runtimes, pd.Series):
        runtimes = runtimes.values.copy()
    else:
        runtimes = runtimes.copy()

    # sort the runtimes
    runtimes.sort()

    # compute expected value
    sum = 0.0
    num_samples = runtimes.shape[0]
    for j in range(1, num_samples+1):
        num = nCr(j-1, k-1) * baseline_runtime
        den = nCr(num_samples, k) * max(runtimes[j-1], 1e-8)
        sum += num / den
    return pd.Series({col_name.format(k): sum})

# def speedupk(df: pd.DataFrame, k: int, n: int, problem_wise: bool = False):
#     """ Compute the speedup@k metric """
#     df = df.copy()

#     # get all runs where is_valid is true
#     df = df[df["is_valid"] == True]

#     # choose processor count; hardcoded right now
#     df = df[(df["parallelism_model"] == "serial") |
#             (df["parallelism_model"] == "cuda") |
#             (df["parallelism_model"] == "hip") |
#             ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
#             ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
#             ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32)) |
#             ((df["parallelism_model"] == "rust") & (df["num_threads"] == 32))]
#     df = df.copy()

#     if df["parallelism_model"] == "rust":
#         df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")
#         df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#             lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k)
#         ).reset_index()
#         # compute the mean speedup@k
#         df = df.groupby(["parallelism_model", "problem_type"]).agg({f"speedup@{k}": "mean"})
#         return df, None
#     else:
#         # Isolate OMP runs to find the baseline for each problem
#         omp_baselines = df[df["parallelism_model"] == "omp"].copy()
#         omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
#         omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

#         # Filter for Parlay runs and merge the OMP baseline
#         parlay_df = df[df["parallelism_model"] == "parlay"].copy()
#         parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')

#         # Drop any rows where an OMP baseline was not found for the problem
#         parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)

#         # Calculate speedup for Parlay using the OMP baseline
#         parlay_results = parlay_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#             lambda row: _speedupk(row["runtime"], np.min(row["omp_baseline_runtime"]), k),
#             include_groups=False
#         ).reset_index()
#         # parlay_results = parlay_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         #     lambda row: _speedupk(row["runtime"], np.min(row["omp_baseline_runtime"]), k)
#         # ).reset_index()

#         # Filter for all other models
#         other_models_df = df[df["parallelism_model"] != "parlay"].copy()

#         # Calculate speedup for other models using their own sequential runtime
#         # other_results = other_models_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         #     lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k)
#         # ).reset_index()
#         other_results = other_models_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#             lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k),
#             include_groups=False
#         ).reset_index()

#         # Combine the two results DataFrames
#         combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

#         # Compute the mean speedup@k for all models
#         if problem_wise:
#             final_df = combined_results.groupby(["parallelism_model", "name"]).agg({f"speedup@{k}": "mean"})
#         else:
#             final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"speedup@{k}": "mean"})


#         # --- New Ratio Calculation ---
#         # Isolate sequential runtimes for Parlay and OMP
#         parlay_seq_df = df[df["parallelism_model"] == "parlay"].groupby("name")["best_sequential_runtime"].min().reset_index()
#         parlay_seq_df.rename(columns={'best_sequential_runtime': 'parlay_seq_runtime'}, inplace=True)
        
#         omp_seq_df = df[df["parallelism_model"] == "omp"].groupby("name")["best_sequential_runtime"].min().reset_index()
#         omp_seq_df.rename(columns={'best_sequential_runtime': 'omp_seq_runtime'}, inplace=True)
        
#         # Merge and calculate the ratio
#         ratio_df = pd.merge(parlay_seq_df, omp_seq_df, on='name', how='inner')
#         ratio_df['parlay_omp_seq_ratio'] = ratio_df['omp_seq_runtime'] / ratio_df['parlay_seq_runtime']

#         # Get problem type for grouping
#         ratio_df = ratio_df.merge(df[["name", "problem_type"]].drop_duplicates(), on="name", how="left")
        
#         if problem_wise:
#             avg_ratio = ratio_df.groupby("name")["parlay_omp_seq_ratio"].mean().reset_index()
#         else:
#             avg_ratio = ratio_df.groupby("problem_type")["parlay_omp_seq_ratio"].mean().reset_index()


#         return final_df, avg_ratio
def speedupk(df: pd.DataFrame, k: int, n: int, problem_wise: bool = False):
    """ Compute the speedup@k metric """
    df = df.copy()

    # 1. Filter Valid Runs
    df = df[df["is_valid"] == True]

    # 2. Filter Models (Include Rust here)
    df = df[(df["parallelism_model"] == "serial") |
            (df["parallelism_model"] == "cuda") |
            (df["parallelism_model"] == "hip") |
            ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "rust") & (df["num_threads"] == 32))]
    df = df.copy()

    # --- Robust Helper Function (Crucial for preventing KeyError) ---
    def process_apply_result(grouped_df, apply_func):
        # group_keys=True ensures keys are in the index (fixes FutureWarning)
        res = grouped_df.apply(apply_func)
        
        # If result is a Series (common in single-column results), convert to DF
        if isinstance(res, pd.Series):
            res = res.to_frame(name=f"speedup@{k}")
        
        # If the expected column is missing
        if f"speedup@{k}" not in res.columns:
            if len(res.columns) == 1:
                # If there's one column, assume it's ours and rename it
                res.columns = [f"speedup@{k}"]
            else:
                # If result is empty, force create the column to prevent crash
                res[f"speedup@{k}"] = pd.Series(dtype=float)
                
        # Select ONLY the metric column
        res = res[[f"speedup@{k}"]]
        return res.reset_index()

    # 3. Handle OMP Baselines (only relevant if OMP exists)
    omp_baselines = df[df["parallelism_model"] == "omp"].copy()
    if not omp_baselines.empty:
        omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
        omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

    # 4. Handle Parlay (Requires OMP baseline)
    parlay_df = df[df["parallelism_model"] == "parlay"].copy()
    if not parlay_df.empty and not omp_baselines.empty:
        parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')
        parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)
        
        parlay_results = process_apply_result(
            parlay_df.groupby(["name", "parallelism_model", "problem_type"], group_keys=True),
            lambda row: _speedupk(row["runtime"], np.min(row["omp_baseline_runtime"]), k)
        )
    else:
        # Create empty DF if no Parlay data
        parlay_results = pd.DataFrame(columns=["name", "parallelism_model", "problem_type", f"speedup@{k}"])

    # 5. Handle Other Models (Rust, CUDA, Kokkos, etc.)
    # This logic automatically works for Rust!
    other_models_df = df[df["parallelism_model"] != "parlay"].copy()
    
    other_results = process_apply_result(
        other_models_df.groupby(["name", "parallelism_model", "problem_type"], group_keys=True),
        lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k)
    )

    # 6. Combine
    combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

    if problem_wise:
        final_df = combined_results.groupby(["parallelism_model", "name"]).agg({f"speedup@{k}": "mean"})
    else:
        final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"speedup@{k}": "mean"})

    # 7. Ratio Calculation (Safe for missing data)
    parlay_seq_df = df[df["parallelism_model"] == "parlay"].groupby("name")["best_sequential_runtime"].min().reset_index()
    parlay_seq_df.rename(columns={'best_sequential_runtime': 'parlay_seq_runtime'}, inplace=True)
    
    omp_seq_df = df[df["parallelism_model"] == "omp"].groupby("name")["best_sequential_runtime"].min().reset_index()
    omp_seq_df.rename(columns={'best_sequential_runtime': 'omp_seq_runtime'}, inplace=True)
    
    ratio_df = pd.merge(parlay_seq_df, omp_seq_df, on='name', how='inner')
    
    if not ratio_df.empty:
        ratio_df['parlay_omp_seq_ratio'] = ratio_df['omp_seq_runtime'] / ratio_df['parlay_seq_runtime']
        ratio_df = ratio_df.merge(df[["name", "problem_type"]].drop_duplicates(), on="name", how="left")
        if problem_wise:
            avg_ratio = ratio_df.groupby("name")["parlay_omp_seq_ratio"].mean().reset_index()
        else:
            avg_ratio = ratio_df.groupby("problem_type")["parlay_omp_seq_ratio"].mean().reset_index()
    else:
        avg_ratio = pd.DataFrame(columns=["problem_type", "parlay_omp_seq_ratio"] if not problem_wise else ["name", "parlay_omp_seq_ratio"])

    return final_df, avg_ratio


def speedupk_max(df: pd.DataFrame, k: int, problem_wise: bool = False) -> pd.DataFrame:
    """ Compute the speedup_max@k. 
        Finds the BEST speedup across all available thread counts for each problem.
    """
    df = df.copy()
    if "prompt" in df.columns:
        df.drop(columns=['prompt'], inplace=True)

    # 1. Filter Valid Runs
    df = df[df["is_valid"] == True]

    # 2. Select the BEST runtime across all available thread counts (True Max Speedup)
    # This finds the min runtime for this specific problem/model combination across all resources
    df["runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["runtime"].transform("min")

    # 3. Standardize Sequential Runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # 4. Remove Duplicates (Since we transformed runtime, we only need one row per problem instance)
    if "run_idx" in df.columns:
        df["run_idx"] = df["run_idx"].astype(int)
        df = df[df["run_idx"] == 0]
    
    # --- Robust Helper Function ---
    def process_apply_result(grouped_df, apply_func):
        # group_keys=True ensures keys are in the index
        res = grouped_df.apply(apply_func)
        
        if isinstance(res, pd.Series):
            res = res.to_frame(name=f"speedup_max@{k}")
        
        # Handle missing columns or empty results
        if f"speedup_max@{k}" not in res.columns:
            if len(res.columns) == 1:
                res.columns = [f"speedup_max@{k}"]
            else:
                res[f"speedup_max@{k}"] = pd.Series(dtype=float)
        
        # CRITICAL: Select ONLY the metric column
        res = res[[f"speedup_max@{k}"]]
        return res.reset_index()

    # 5. Handle OMP Baselines (Fair Comparison for Parlay)
    omp_baselines = df[df["parallelism_model"] == "omp"].copy()
    if not omp_baselines.empty:
        omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
        omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

    # 6. Handle Parlay (Merge OMP Baseline)
    parlay_df = df[df["parallelism_model"] == "parlay"].copy()
    if not parlay_df.empty and not omp_baselines.empty:
        parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')
        parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)
        
        parlay_results = process_apply_result(
            parlay_df.groupby(["name", "parallelism_model", "problem_type"], group_keys=True),
            lambda row: _speedupk(row["runtime"], np.min(row["omp_baseline_runtime"]), k, col_name=f"speedup_max@{k}")
        )
    else:
        parlay_results = pd.DataFrame(columns=["name", "parallelism_model", "problem_type", f"speedup_max@{k}"])

    # 7. Handle Other Models (Use their own sequential baseline)
    other_models_df = df[df["parallelism_model"] != "parlay"].copy()
    
    other_results = process_apply_result(
        other_models_df.groupby(["name", "parallelism_model", "problem_type"], group_keys=True),
        lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k, col_name=f"speedup_max@{k}")
    )

    # 8. Combine Results
    combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

    # 9. Aggregate (Support problem_wise)
    if problem_wise:
        final_df = combined_results.groupby(["parallelism_model", "name"]).agg({f"speedup_max@{k}": "mean"})
    else:
        final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"speedup_max@{k}": "mean"})

    return final_df

def parlay_omp_sequantial_runtime_ratio(df: pd.DataFrame, k: int, problem_wise: bool = False) -> pd.DataFrame:
    """ Compute the speedup@k metric for Parlay using OpenMP's sequential baseline. """
    df = df.copy()
    
    df = df[(df["parallelism_model"] == "serial") |
            (df["parallelism_model"] == "cuda") |
            (df["parallelism_model"] == "hip") |
            ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32))]
    df = df.copy()

    # --- New Ratio Calculation ---
    # Isolate sequential runtimes for Parlay and OMP
    parlay_seq_df = df[df["parallelism_model"] == "parlay"].groupby("name")["best_sequential_runtime"].min().reset_index()
    parlay_seq_df.rename(columns={'best_sequential_runtime': 'parlay_seq_runtime'}, inplace=True)
    
    omp_seq_df = df[df["parallelism_model"] == "omp"].groupby("name")["best_sequential_runtime"].min().reset_index()
    omp_seq_df.rename(columns={'best_sequential_runtime': 'omp_seq_runtime'}, inplace=True)
    
    # Merge and calculate the ratio
    ratio_df = pd.merge(parlay_seq_df, omp_seq_df, on='name', how='inner')
    ratio_df['parlay_omp_seq_ratio'] = ratio_df['omp_seq_runtime'] / ratio_df['parlay_seq_runtime'] 

    # Get problem type for grouping
    ratio_df = ratio_df.merge(df[["name", "problem_type"]].drop_duplicates(), on="name", how="left")
    
    if problem_wise:
        avg_ratio = ratio_df.groupby("name")["parlay_omp_seq_ratio"].mean().reset_index()
    else:
        avg_ratio = ratio_df.groupby("problem_type")["parlay_omp_seq_ratio"].mean().reset_index()

    # Add the new ratio as a column to the final DataFrame
    final_df = pd.merge(df, avg_ratio, on="problem_type", how="left")
  
    return final_df

def _efficiencyk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int, n_resources: Union[pd.Series, np.ndarray], col_name: str = 'efficiency@{}') -> float:
    """ Compute the efficiency@k metric """
    # create a copy of the runtimes
    if isinstance(runtimes, pd.Series):
        runtimes = runtimes.values.copy()
    else:
        runtimes = runtimes.copy()

    if isinstance(n_resources, pd.Series):
        n_resources = n_resources.values.copy()
    else:
        n_resources = n_resources.copy()

    # sort the runtimes
    runtimes.sort()

    # compute expected value
    sum = 0.0
    num_samples = runtimes.shape[0]
    for j in range(1, num_samples+1):
        num = nCr(j-1, k-1) * baseline_runtime
        den = nCr(num_samples, k) * max(runtimes[j-1], 1e-8) * n_resources[j-1]
        sum += num / den
    return pd.Series({col_name.format(k): sum})

# def efficiencyk(df: pd.DataFrame, k: int, n: int, problem_wise: bool = False) -> pd.DataFrame:
#     """ Compute the efficiency@k metric """
#     df = df.copy()

#     # get all runs where is_valid is true
#     df = df[df["is_valid"] == True]

#     # choose processor count; hardcoded right now
#     df = df[(df["parallelism_model"] == "serial") |
#         (df["parallelism_model"] == "cuda") |
#         (df["parallelism_model"] == "hip") |
#         ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
#         ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
#         ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32)) | 
#         ((df["parallelism_model"] == "rust") & (df["num_threads"] == 32))]

#     # set n_resources column to 1 for serial; 32 for kokkos; 32 for omp; 512 for mpi; 4*64 for mpi+omp;
#     # set it to problem_size for cuda and hip
#     df["n_resources"] = 1
#     df.loc[df["parallelism_model"] == "cuda", "n_resources"] = df["problem_size"]
#     df.loc[df["parallelism_model"] == "hip", "n_resources"] = df["problem_size"]
#     df.loc[df["parallelism_model"] == "kokkos", "n_resources"] = 32
#     df.loc[df["parallelism_model"] == "omp", "n_resources"] = 8
#     df.loc[df["parallelism_model"] == "parlay", "n_resources"] = 8
#     df.loc[df["parallelism_model"] == "mpi", "n_resources"] = 512
#     df.loc[df["parallelism_model"] == "mpi+omp", "n_resources"] = 4*64
#     df.loc[df["parallelism_model"] == "rust", "n_resources"] = 8

#     df = df.copy()

#     # use min best_sequential_runtime
#     df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

#     # --- New Logic for Parlay and Other Models ---
#     # Isolate OMP runs to find the baseline for each problem
#     omp_baselines = df[df["parallelism_model"] == "omp"].copy()
#     omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
#     omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

#     # Filter for Parlay runs and merge the OMP baseline
#     parlay_df = df[df["parallelism_model"] == "parlay"].copy()
#     parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')
#     parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)

#     # Calculate speedup_max for Parlay using the OMP baseline
#     parlay_results = parlay_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _efficiencyk(row["runtime"], np.min(row["omp_baseline_runtime"]), k, row["n_resources"])
#     ).reset_index()

#     # Filter for all other models
#     other_models_df = df[df["parallelism_model"] != "parlay"].copy()

#     # Calculate speedup_max for other models using their own sequential runtime
#     other_results = other_models_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"])
#     ).reset_index()

#     # Combine the two results DataFrames
#     combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

#     # compute the mean speedup_max@k
#     if problem_wise:
#         final_df = combined_results.groupby(["parallelism_model", "name"]).agg({f"efficiency@{k}": "mean"})
#     else:
#         final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency@{k}": "mean"})

#     return final_df

def efficiencyk(df: pd.DataFrame, k: int, n: int, problem_wise: bool = False) -> pd.DataFrame:
    """ Compute the efficiency@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # choose processor count
    df = df[(df["parallelism_model"] == "serial") |
        (df["parallelism_model"] == "cuda") |
        (df["parallelism_model"] == "hip") |
        ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
        ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
        ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32)) | 
        ((df["parallelism_model"] == "rust") & (df["num_threads"] == 32))]

    # set n_resources column
    df["n_resources"] = 1
    df.loc[df["parallelism_model"] == "cuda", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "hip", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "kokkos", "n_resources"] = 32
    df.loc[df["parallelism_model"] == "omp", "n_resources"] = 8
    df.loc[df["parallelism_model"] == "parlay", "n_resources"] = 8
    df.loc[df["parallelism_model"] == "mpi", "n_resources"] = 512
    df.loc[df["parallelism_model"] == "mpi+omp", "n_resources"] = 4*64
    df.loc[df["parallelism_model"] == "rust", "n_resources"] = 8

    df = df.copy()

    # use min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # --- Robust Helper Function ---
    def process_apply_result(grouped_df, apply_func):
        res = grouped_df.apply(apply_func)
        
        if isinstance(res, pd.Series):
            res = res.to_frame(name=f"efficiency@{k}")
        
        # Handle missing columns or empty results
        if f"efficiency@{k}" not in res.columns:
            if len(res.columns) == 1:
                res.columns = [f"efficiency@{k}"]
            else:
                res[f"efficiency@{k}"] = pd.Series(dtype=float)
                
        # CRITICAL FIX: Select ONLY the metric column to prevent 'ValueError: cannot insert... already exists'
        res = res[[f"efficiency@{k}"]]
        return res.reset_index()

    # Isolate OMP runs to find the baseline for each problem
    omp_baselines = df[df["parallelism_model"] == "omp"].copy()
    if not omp_baselines.empty:
        omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
        omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

    # Filter for Parlay runs and merge the OMP baseline
    parlay_df = df[df["parallelism_model"] == "parlay"].copy()
    if not parlay_df.empty and not omp_baselines.empty:
        parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')
        parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)

        parlay_results = process_apply_result(
            parlay_df.groupby(["name", "parallelism_model", "problem_type"], group_keys=True),
            lambda row: _efficiencyk(row["runtime"], np.min(row["omp_baseline_runtime"]), k, row["n_resources"])
        )
    else:
        parlay_results = pd.DataFrame(columns=["name", "parallelism_model", "problem_type", f"efficiency@{k}"])

    # Filter for all other models
    other_models_df = df[df["parallelism_model"] != "parlay"].copy()

    # Calculate efficiency for other models using their own sequential runtime
    other_results = process_apply_result(
        other_models_df.groupby(["name", "parallelism_model", "problem_type"], group_keys=True),
        lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"])
    )

    # Combine the two results DataFrames
    combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

    # compute the mean efficiency@k
    if problem_wise:
        final_df = combined_results.groupby(["parallelism_model", "name"]).agg({f"efficiency@{k}": "mean"})
    else:
        final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency@{k}": "mean"})

    return final_df

def efficiencyk_max(df: pd.DataFrame, k: int, problem_wise: bool = False) -> pd.DataFrame:
    """ Compute the efficiency_max@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # set n_resources column
    df["n_resources"] = 1
    df.loc[df["parallelism_model"] == "cuda", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "hip", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "kokkos", "n_resources"] = df["num_threads"]
    df.loc[df["parallelism_model"] == "omp", "n_resources"] = df["num_threads"]
    df.loc[df["parallelism_model"] == "parlay", "n_resources"] = df["num_threads"]
    df.loc[df["parallelism_model"] == "rust", "n_resources"] = df["num_threads"]

    # choose the row with min num_resources * runtime
    # Using group_keys=False here is safe as we reset index immediately after
    df = df.groupby(["name", "parallelism_model", "output_idx"], group_keys=False).apply(
            lambda row: row.iloc[np.argmin(row["runtime"] * row["n_resources"])]
        ).reset_index(drop=True)

    # use the min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # --- Robust Helper Function ---
    def process_apply_result(grouped_df, apply_func):
        res = grouped_df.apply(apply_func)
        
        if isinstance(res, pd.Series):
            res = res.to_frame(name=f"efficiency_max@{k}")
        
        if f"efficiency_max@{k}" not in res.columns:
            if len(res.columns) == 1:
                res.columns = [f"efficiency_max@{k}"]
            else:
                res[f"efficiency_max@{k}"] = pd.Series(dtype=float)
                
        # CRITICAL FIX: Select ONLY the metric column
        res = res[[f"efficiency_max@{k}"]]
        return res.reset_index()

    # Isolate OMP runs to find the baseline for each problem
    omp_baselines = df[df["parallelism_model"] == "omp"].copy()
    if not omp_baselines.empty:
        omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
        omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

    # Filter for Parlay runs and merge the OMP baseline
    parlay_df = df[df["parallelism_model"] == "parlay"].copy()
    if not parlay_df.empty and not omp_baselines.empty:
        parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')
        parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)

        parlay_results = process_apply_result(
            parlay_df.groupby(["name", "parallelism_model", "problem_type"], group_keys=True),
            lambda row: _efficiencyk(row["runtime"], np.min(row["omp_baseline_runtime"]), k, row["n_resources"], col_name='efficiency_max@{}')
        )
    else:
        parlay_results = pd.DataFrame(columns=["name", "parallelism_model", "problem_type", f"efficiency_max@{k}"])

    # Filter for all other models
    other_models_df = df[df["parallelism_model"] != "parlay"].copy()

    # Calculate efficiency for other models using their own sequential runtime
    other_results = process_apply_result(
        other_models_df.groupby(["name", "parallelism_model", "problem_type"], group_keys=True),
        lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"], col_name='efficiency_max@{}')
    )

    # Combine the two results DataFrames
    combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

    # compute the mean efficiency@k
    if problem_wise:
        final_df = combined_results.groupby(["parallelism_model", "name"]).agg({f"efficiency_max@{k}": "mean"})
    else:
        final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency_max@{k}": "mean"})

    return final_df

# def efficiencyk_max(df: pd.DataFrame, k: int, problem_wise: bool = False) -> pd.DataFrame:
#     """ Compute the efficiency_max@k metric """
#     df = df.copy()

#     # get all runs where is_valid is true
#     df = df[df["is_valid"] == True]

#     # set n_resources column
#     df["n_resources"] = 1
#     df.loc[df["parallelism_model"] == "cuda", "n_resources"] = df["problem_size"]
#     df.loc[df["parallelism_model"] == "hip", "n_resources"] = df["problem_size"]
#     df.loc[df["parallelism_model"] == "kokkos", "n_resources"] = df["num_threads"]
#     df.loc[df["parallelism_model"] == "omp", "n_resources"] = df["num_threads"]
#     df.loc[df["parallelism_model"] == "parlay", "n_resources"] = df["num_threads"]
#     df.loc[df["parallelism_model"] == "rust", "n_resources"] = df["num_threads"]

#     # choose the row with min num_resources * runtime
#     df = df.groupby(["name", "parallelism_model", "output_idx"]).apply(
#             lambda row: row.iloc[np.argmin(row["runtime"] * row["n_resources"])]
#         ).reset_index(drop=True)

#     # use the min best_sequential_runtime
#     df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

#     # --- New Logic for Parlay and Other Models ---
#     # Isolate OMP runs to find the baseline for each problem
#     omp_baselines = df[df["parallelism_model"] == "omp"].copy()
#     omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
#     omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

#     # Filter for Parlay runs and merge the OMP baseline
#     parlay_df = df[df["parallelism_model"] == "parlay"].copy()
#     parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')
#     parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)

#     # Calculate speedup_max for Parlay using the OMP baseline
#     parlay_results = parlay_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _efficiencyk(row["runtime"], np.min(row["omp_baseline_runtime"]), k, row["n_resources"], col_name='efficiency_max@{}')
#     ).reset_index()

#     # Filter for all other models
#     other_models_df = df[df["parallelism_model"] != "parlay"].copy()

#     # Calculate speedup_max for other models using their own sequential runtime
#     other_results = other_models_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"], col_name='efficiency_max@{}')
#     ).reset_index()

#     # Combine the two results DataFrames
#     combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

#     # compute the mean speedup_max@k
#     if problem_wise:
#         final_df = combined_results.groupby(["parallelism_model", "name"]).agg({f"efficiency_max@{k}": "mean"})
#     else:
#         final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency_max@{k}": "mean"})

#     return final_df

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

    # get only valid runs
    valid_runs = get_correctness_df(df)
    
    # get values for each k (problem-type-wise)
    all_results = []
    avg_ratio = None
    for k in args.k:
        build_values = buildk(df, k)
        pass_values = passk(valid_runs, k)
        speedup_values, current_avg_ratio = speedupk(df, k, args.n)
        if current_avg_ratio is not None:
            avg_ratio = current_avg_ratio
        # save_fastest_parlay_omp(df, k, args.n, f"fastest_parlayscheduler_omp_k{k}.json")
        speedup_max_values = speedupk_max(df, k)
        efficiency_values = efficiencyk(df, k, args.n)
        efficiency_max_values = efficiencyk_max(df, k)
        all_results.extend([build_values, pass_values, speedup_values, speedup_max_values, efficiency_values, efficiency_max_values])
    
    merged_df = pd.concat(all_results, axis=1).reset_index()
    if avg_ratio is not None:
        merged_df = pd.merge(merged_df, avg_ratio, on="problem_type", how="left")

    # if there were no successfull builds or runs, then speedup@k will be nan after merging
    # replace NaN speedup@k values with 0.0
    for k in args.k:
        merged_df[f"speedup@{k}"] = merged_df[f"speedup@{k}"].fillna(0.0)
        merged_df[f"speedup_max@{k}"] = merged_df[f"speedup_max@{k}"].fillna(0.0)
        merged_df[f"efficiency@{k}"] = merged_df[f"efficiency@{k}"].fillna(0.0)
        merged_df[f"efficiency_max@{k}"] = merged_df[f"efficiency_max@{k}"].fillna(0.0)

    # add model name column
    if args.model_name:
        merged_df.insert(0, "model_name", args.model_name)

    # clean up column names
    column_name_map = {
        "model_name": "model",
        "parallelism_model": "execution model",
        "problem_type": "problem type",
    }
    merged_df = merged_df.rename(columns=column_name_map)

    # write to csv
    if args.output:
        merged_df.to_csv(args.output, index=False)
    else:
        pd.set_option('display.max_columns', merged_df.shape[1]+1)
        pd.set_option('display.max_rows', merged_df.shape[0]+1)
        print("Problem-Type-wise Metrics:")
        print(merged_df)

    # Now, handle problem-wise metrics if the flag is set
    if args.problem_wise:
        all_results_problem_wise = []
        avg_ratio_problem_wise = None
        for k in args.k:
            build_values_pw = buildk(df, k, problem_wise=True)
            pass_values_pw = passk(valid_runs, k, problem_wise=True)
            speedup_values_pw, current_avg_ratio_pw = speedupk(df, k, args.n, problem_wise=True)
            if current_avg_ratio_pw is not None:
                avg_ratio_problem_wise = current_avg_ratio_pw
            speedup_max_values_pw = speedupk_max(df, k, problem_wise=True)
            efficiency_values_pw = efficiencyk(df, k, args.n, problem_wise=True)
            efficiency_max_values_pw = efficiencyk_max(df, k, problem_wise=True)
            all_results_problem_wise.extend([build_values_pw, pass_values_pw, speedup_values_pw, speedup_max_values_pw, efficiency_values_pw, efficiency_max_values_pw])

        merged_df_problem_wise = pd.concat(all_results_problem_wise, axis=1).reset_index()
        if avg_ratio_problem_wise is not None:
            merged_df_problem_wise = pd.merge(merged_df_problem_wise, avg_ratio_problem_wise, on="name", how="left")
            
        # replace NaN speedup@k values with 0.0
        for k in args.k:
            merged_df_problem_wise[f"speedup@{k}"] = merged_df_problem_wise[f"speedup@{k}"].fillna(0.0)
            merged_df_problem_wise[f"speedup_max@{k}"] = merged_df_problem_wise[f"speedup_max@{k}"].fillna(0.0)
            merged_df_problem_wise[f"efficiency@{k}"] = merged_df_problem_wise[f"efficiency@{k}"].fillna(0.0)
            merged_df_problem_wise[f"efficiency_max@{k}"] = merged_df_problem_wise[f"efficiency_max@{k}"].fillna(0.0)

        if args.model_name:
            merged_df_problem_wise.insert(0, "model_name", args.model_name)
        
        column_name_map_pw = {
            "model_name": "model",
            "parallelism_model": "execution model",
            "name": "problem name",
        }
        merged_df_problem_wise = merged_df_problem_wise.rename(columns=column_name_map_pw)

        if args.output:
            output_file_name, output_ext = args.output.rsplit('.', 1)
            merged_df_problem_wise.to_csv(f"{output_file_name}_problem_wise.{output_ext}", index=False)
        else:
            print("\nProblem-wise Metrics:")
            print(merged_df_problem_wise)
            
if __name__ == "__main__":
    main()


# """ Compute the metrics over the data.
# """
# # std imports
# import argparse
# import json
# from math import comb
# from typing import Union

# # tpl imports
# import numpy as np
# import pandas as pd


# def get_args():
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("input_csv", type=str, help="Input CSV file containing the test cases.")
#     parser.add_argument("-k", "--k", type=int, nargs='+', default=[1,5,10,20], help="K value for pass@k, build@k, and speedup@k.")
#     parser.add_argument("-n", "--n", type=int, default=1, help="N value for speedup@k.")
#     parser.add_argument("-o", "--output", type=str, help="Output csv file containing the results.")
#     parser.add_argument("--problem-sizes", type=str, default='../drivers/problem-sizes.json', help="Json with problem sizes. Used for calculating GPU efficiency.")
#     parser.add_argument("--model-name", type=str, help="Add model name column with this value")
#     parser.add_argument("--problem-wise", action="store_true", help="If turned on, generate problem-wise metrics in addition to problem-type-wise metrics.")
#     return parser.parse_args()

# def get_correctness_df(df: pd.DataFrame) -> pd.DataFrame:
#     """ Group by name, parallelism_model, and output_idx, and set is_valid to true only if all rows in the group have is_valid = true.
#         Set it to false otherwise.
#     """
#     # group all the runs for this LLM output
#     df = df.copy()
#     agg = df.groupby(["name", "parallelism_model", "output_idx"]).agg({"is_valid": ["count", "sum"]})
#     agg.columns = ["count", "sum"]

#     # mark as valid only if all runs are valid
#     agg["is_valid"] = agg["count"] == agg["sum"]
#     agg = agg.reset_index()
#     agg = agg.drop(columns=["count", "sum"])
    
#     # add problem_type column from df
#     agg = agg.merge(df[["name", "problem_type"]].drop_duplicates(), on="name", how="left")

#     return agg

# def nCr(n: int, r: int) -> int:
#     if n < r:
#         return 1
#     return comb(n, r)

# def buildk(df: pd.DataFrame, k: int, problem_wise: bool = False) -> pd.DataFrame:
#     """ Compute the build@k metric """
#     agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg({"did_build": ["count", "sum"]})
#     agg.columns = ["total_build_attempts", "successful_builds"]
#     agg = agg.reset_index()
#     agg[f"build@{k}"] = agg.apply(lambda x: _passk(x["total_build_attempts"], x["successful_builds"], k), axis=1)
#     if problem_wise:
#         return agg.groupby(["parallelism_model", "name"]).agg({f"build@{k}": "mean"})
#     else:
#         return agg.groupby(["parallelism_model", "problem_type"]).agg({f"build@{k}": "mean"})

# def _passk(num_samples: int, num_correct: int, k: int) -> float:
#     if num_samples - num_correct < k:
#         return 1.0
#     return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))

# def passk(df: pd.DataFrame, k: int, problem_wise: bool = False) -> pd.DataFrame:
#     """ Compute the pass@k metric """
#     agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg({"is_valid": ["count", "sum"]})
#     agg.columns = ["total_runs", "valid_count"]
#     agg = agg.reset_index()
#     agg[f"pass@{k}"] = agg.apply(lambda x: _passk(x["total_runs"], x["valid_count"], k), axis=1)
#     if problem_wise:
#         return agg.groupby(["parallelism_model", "name"]).agg({f"pass@{k}": "mean"})
#     return agg.groupby(["parallelism_model", "problem_type"]).agg({f"pass@{k}": "mean"})

# def _speedupk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int, col_name: str = 'speedup@{}') -> float:
#     """ Compute the speedup@k metric """
#     # create a copy of the runtimes
#     if isinstance(runtimes, pd.Series):
#         runtimes = runtimes.values.copy()
#     else:
#         runtimes = runtimes.copy()

#     # sort the runtimes
#     runtimes.sort()

#     # compute expected value
#     sum = 0.0
#     num_samples = runtimes.shape[0]
#     for j in range(1, num_samples+1):
#         num = nCr(j-1, k-1) * baseline_runtime
#         den = nCr(num_samples, k) * max(runtimes[j-1], 1e-8)
#         sum += num / den
#     return pd.Series({col_name.format(k): sum})

# def speedupk(df: pd.DataFrame, k: int, n: int):
#     """ Compute the speedup@k metric """
#     df = df.copy()

#     # get all runs where is_valid is true
#     df = df[df["is_valid"] == True]

#     # choose processor count; hardcoded right now
#     # df = df[(df["parallelism_model"] == "serial") |
#     #         (df["parallelism_model"] == "cuda") |
#     #         (df["parallelism_model"] == "hip") |
#     #         ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
#     #         ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
#     #         ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32)) |
#     #         ((df["parallelism_model"] == "mpi") & (df["num_procs"] == 512)) |
#     #         ((df["parallelism_model"] == "mpi+omp") & (df["num_procs"] == 4) & (df["num_threads"] == 64))]
    
#     df = df[(df["parallelism_model"] == "serial") |
#             (df["parallelism_model"] == "cuda") |
#             (df["parallelism_model"] == "hip") |
#             ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
#             ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
#             ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32))]
#     df = df.copy()

#     # Isolate OMP runs to find the baseline for each problem
#     omp_baselines = df[df["parallelism_model"] == "omp"].copy()
#     omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
#     omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

#     # Filter for Parlay runs and merge the OMP baseline
#     parlay_df = df[df["parallelism_model"] == "parlay"].copy()
#     parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')

#     # Drop any rows where an OMP baseline was not found for the problem
#     parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)

#     # Calculate speedup for Parlay using the OMP baseline
#     parlay_results = parlay_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _speedupk(row["runtime"], np.min(row["omp_baseline_runtime"]), k)
#     ).reset_index()

#     # Filter for all other models
#     other_models_df = df[df["parallelism_model"] != "parlay"].copy()

#     # Calculate speedup for other models using their own sequential runtime
#     other_results = other_models_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k)
#     ).reset_index()

#     # Combine the two results DataFrames
#     combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

#     # Compute the mean speedup@k for all models
#     final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"speedup@{k}": "mean"})

#     # --- New Ratio Calculation ---
#     # Isolate sequential runtimes for Parlay and OMP
#     parlay_seq_df = df[df["parallelism_model"] == "parlay"].groupby("name")["best_sequential_runtime"].min().reset_index()
#     parlay_seq_df.rename(columns={'best_sequential_runtime': 'parlay_seq_runtime'}, inplace=True)
    
#     omp_seq_df = df[df["parallelism_model"] == "omp"].groupby("name")["best_sequential_runtime"].min().reset_index()
#     omp_seq_df.rename(columns={'best_sequential_runtime': 'omp_seq_runtime'}, inplace=True)
    
#     # Merge and calculate the ratio
#     ratio_df = pd.merge(parlay_seq_df, omp_seq_df, on='name', how='inner')
#     ratio_df['parlay_omp_seq_ratio'] = ratio_df['omp_seq_runtime'] / ratio_df['parlay_seq_runtime']

#     # Get problem type for grouping
#     ratio_df = ratio_df.merge(df[["name", "problem_type"]].drop_duplicates(), on="name", how="left")
#     avg_ratio = ratio_df.groupby("problem_type")["parlay_omp_seq_ratio"].mean().reset_index()

#     # Add the new ratio as a column to the final DataFrame
#     # Note: This adds a summary metric for each problem type, not a per-row value.
#     # final_df = pd.merge(final_df, avg_ratio, on="problem_type", how="left")

#     # # use min best_sequential_runtime
#     # df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

#     # # group by name, parallelism_model, and output_idx and call _speedupk
#     # df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#     #         lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k)
#     #     ).reset_index()

#     # # compute the mean speedup@k
#     # df = df.groupby(["parallelism_model", "problem_type"]).agg({f"speedup@{k}": "mean"})

#     return final_df, avg_ratio

# def save_fastest_parlay_omp(df: pd.DataFrame, k: int, n: int, output_file: str):
#     """
#     Finds the fastest Parlay and OMP implementation for each problem
#     and saves the corresponding code to a JSON file.
#     """
#     # The original speedupk function can still be used for data preparation.
#     # It returns the filtered df and the ratio_df which we can use for finding fastest run.
#     # It also has the filtering logic that we need, so let's reuse it.
#     df = df.copy()

#     # get all runs where is_valid is true
#     df = df[df["is_valid"] == True]

#     # final_df, avg_ratio = speedupk(df, k, n)

#     # Filter the original DataFrame for just Parlay and OMP
#     parlay_omp_df = df[df["parallelism_model"].isin(["parlay", "omp"])].copy()

#     # Find the indices of the fastest run for each group.
#     # We use .dropna() to prevent KeyErrors from empty groups.
#     fastest_indices = parlay_omp_df.groupby(["name", "parallelism_model"])["runtime"].idxmin().dropna()

#     # Use the indices to select the corresponding rows from the DataFrame.
#     fastest_runs = parlay_omp_df.loc[fastest_indices].reset_index(drop=True)
#     print("fastest_runs:", fastest_runs['generated_output'])


#     # Select the columns to be saved
#     output_data = fastest_runs[[
#         "name",
#         "problem_type",
#         "parallelism_model",
#         "runtime",
#         "generated_output"
#     ]].to_dict(orient="records")

#     # Save as JSON
#     with open(output_file, "w") as f:
#         json.dump(output_data, f, indent=2)

#     print(f"Fastest implementations for Parlay and OMP saved to {output_file}")
  

# def speedupk_max(df: pd.DataFrame, k: int) -> pd.DataFrame:
#     """ Compute the speedup_max@k. Same as speedup_n@k, but instead of a fixed n
#         we use the n that gives the max speedup
#     """
#     df = df.copy()
#     df.drop(columns=['prompt'], inplace=True)

#     # get all the runs where the submission is valid
#     df = df[df["is_valid"] == True]

#     # choose the min across processor counts
#     df["runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["runtime"].transform("min")

#     # use the min best_sequential_runtime
#     df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

#     # select only run_idx 0
#     df["run_idx"] = df["run_idx"].astype(int)
#     df = df[df["run_idx"] == 0]

#     # # group by name, parallelism_model, and output_idx and call _speedupk
#     # df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#     #         lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k, col_name="speedup_max@{}")
#     #     ).reset_index()

#     # # compute the mean speedup_max@k
#     # df = df.groupby(["parallelism_model", "problem_type"]).agg({f"speedup_max@{k}": "mean"})

#     # --- New Logic for Parlay and Other Models ---
#     # Isolate OMP runs to find the baseline for each problem
#     omp_baselines = df[df["parallelism_model"] == "omp"].copy()
#     omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
#     omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

#     # Filter for Parlay runs and merge the OMP baseline
#     parlay_df = df[df["parallelism_model"] == "parlay"].copy()
#     parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')
#     parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)

#     # Calculate speedup_max for Parlay using the OMP baseline
#     parlay_results = parlay_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _speedupk(row["runtime"], np.min(row["omp_baseline_runtime"]), k, col_name="speedup_max@{}")
#     ).reset_index()

#     # Filter for all other models
#     other_models_df = df[df["parallelism_model"] != "parlay"].copy()

#     # Calculate speedup_max for other models using their own sequential runtime
#     other_results = other_models_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k, col_name="speedup_max@{}")
#     ).reset_index()

#     # Combine the two results DataFrames
#     combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

#     # compute the mean speedup_max@k
#     final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"speedup_max@{k}": "mean"})

#     return final_df

# def parlay_omp_sequantial_runtime_ratio(df: pd.DataFrame, k: int) -> pd.DataFrame:
#     """ Compute the speedup@k metric for Parlay using OpenMP's sequential baseline. """
#     df = df.copy()

#     # get all runs where is_valid is true
#     # df = df[df["is_valid"] == True]

#     # choose processor count; hardcoded right now
#     # df = df[(df["parallelism_model"] == "serial") |
#     #         (df["parallelism_model"] == "cuda") |
#     #         (df["parallelism_model"] == "hip") |
#     #         ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
#     #         ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
#     #         ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32)) |
#     #         ((df["parallelism_model"] == "mpi") & (df["num_procs"] == 512)) |
#     #         ((df["parallelism_model"] == "mpi+omp") & (df["num_procs"] == 4) & (df["num_threads"] == 64))]
    
#     df = df[(df["parallelism_model"] == "serial") |
#             (df["parallelism_model"] == "cuda") |
#             (df["parallelism_model"] == "hip") |
#             ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
#             ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
#             ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32))]
#     df = df.copy()

#     # --- New Ratio Calculation ---
#     # Isolate sequential runtimes for Parlay and OMP
#     parlay_seq_df = df[df["parallelism_model"] == "parlay"].groupby("name")["best_sequential_runtime"].min().reset_index()
#     parlay_seq_df.rename(columns={'best_sequential_runtime': 'parlay_seq_runtime'}, inplace=True)
    
#     omp_seq_df = df[df["parallelism_model"] == "omp"].groupby("name")["best_sequential_runtime"].min().reset_index()
#     omp_seq_df.rename(columns={'best_sequential_runtime': 'omp_seq_runtime'}, inplace=True)
    
#     # Merge and calculate the ratio
#     ratio_df = pd.merge(parlay_seq_df, omp_seq_df, on='name', how='inner')
#     ratio_df['parlay_omp_seq_ratio'] = ratio_df['omp_seq_runtime'] / ratio_df['parlay_seq_runtime'] 

#     # Get problem type for grouping
#     ratio_df = ratio_df.merge(df[["name", "problem_type"]].drop_duplicates(), on="name", how="left")
#     avg_ratio = ratio_df.groupby("problem_type")["parlay_omp_seq_ratio"].mean().reset_index()

#     # Add the new ratio as a column to the final DataFrame
#     # Note: This adds a summary metric for each problem type, not a per-row value.
#     final_df = pd.merge(df, avg_ratio, on="problem_type", how="left")
  
#     return final_df

# def _efficiencyk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int, n_resources: Union[pd.Series, np.ndarray], col_name: str = 'efficiency@{}') -> float:
#     """ Compute the efficiency@k metric """
#     # create a copy of the runtimes
#     if isinstance(runtimes, pd.Series):
#         runtimes = runtimes.values.copy()
#     else:
#         runtimes = runtimes.copy()

#     if isinstance(n_resources, pd.Series):
#         n_resources = n_resources.values.copy()
#     else:
#         n_resources = n_resources.copy()

#     # sort the runtimes
#     runtimes.sort()

#     # compute expected value
#     sum = 0.0
#     num_samples = runtimes.shape[0]
#     for j in range(1, num_samples+1):
#         num = nCr(j-1, k-1) * baseline_runtime
#         den = nCr(num_samples, k) * max(runtimes[j-1], 1e-8) * n_resources[j-1]
#         sum += num / den
#     return pd.Series({col_name.format(k): sum})

# def efficiencyk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
#     """ Compute the efficiency@k metric """
#     df = df.copy()

#     # get all runs where is_valid is true
#     df = df[df["is_valid"] == True]

#     # choose processor count; hardcoded right now
#     # df = df[(df["parallelism_model"] == "serial") |
#     #        (df["parallelism_model"] == "cuda") |
#     #         (df["parallelism_model"] == "hip") |
#     #         ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
#     #         ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
#     #         ((df["parallelism_model"] == "mpi") & (df["num_procs"] == 512)) |
#     #         ((df["parallelism_model"] == "mpi+omp") & (df["num_procs"] == 4) & (df["num_threads"] == 64))]
    
#     df = df[(df["parallelism_model"] == "serial") |
#         (df["parallelism_model"] == "cuda") |
#         (df["parallelism_model"] == "hip") |
#         ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
#         ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
#         ((df["parallelism_model"] == "parlay") & (df["num_threads"] == 32))]

#     # set n_resources column to 1 for serial; 32 for kokkos; 32 for omp; 512 for mpi; 4*64 for mpi+omp;
#     # set it to problem_size for cuda and hip
#     df["n_resources"] = 1
#     df.loc[df["parallelism_model"] == "cuda", "n_resources"] = df["problem_size"]
#     df.loc[df["parallelism_model"] == "hip", "n_resources"] = df["problem_size"]
#     df.loc[df["parallelism_model"] == "kokkos", "n_resources"] = 32
#     df.loc[df["parallelism_model"] == "omp", "n_resources"] = 8
#     df.loc[df["parallelism_model"] == "parlay", "n_resources"] = 8
#     df.loc[df["parallelism_model"] == "mpi", "n_resources"] = 512
#     df.loc[df["parallelism_model"] == "mpi+omp", "n_resources"] = 4*64

#     df = df.copy()

#     # use min best_sequential_runtime
#     df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

#     # --- New Logic for Parlay and Other Models ---
#     # Isolate OMP runs to find the baseline for each problem
#     omp_baselines = df[df["parallelism_model"] == "omp"].copy()
#     omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
#     omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

#     # Filter for Parlay runs and merge the OMP baseline
#     parlay_df = df[df["parallelism_model"] == "parlay"].copy()
#     parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')
#     parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)

#     # Calculate speedup_max for Parlay using the OMP baseline
#     parlay_results = parlay_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _efficiencyk(row["runtime"], np.min(row["omp_baseline_runtime"]), k, row["n_resources"])
#     ).reset_index()

#     # Filter for all other models
#     other_models_df = df[df["parallelism_model"] != "parlay"].copy()

#     # Calculate speedup_max for other models using their own sequential runtime
#     other_results = other_models_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"])
#     ).reset_index()

#     # Combine the two results DataFrames
#     combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

#     # compute the mean speedup_max@k
#     final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency@{k}": "mean"})

#     # # group by name, parallelism_model, and output_idx and call _efficiencyk
#     # df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#     #         lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"])
#     #     ).reset_index()
    
#     # # compute the mean efficiency@k
#     # df = df.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency@{k}": "mean"})

#     return final_df


# def efficiencyk_max(df: pd.DataFrame, k: int) -> pd.DataFrame:
#     """ Compute the efficiency_max@k metric """
#     df = df.copy()

#     # get all runs where is_valid is true
#     df = df[df["is_valid"] == True]

#     # set n_resources column
#     df["n_resources"] = 1
#     df.loc[df["parallelism_model"] == "cuda", "n_resources"] = df["problem_size"]
#     df.loc[df["parallelism_model"] == "hip", "n_resources"] = df["problem_size"]
#     df.loc[df["parallelism_model"] == "kokkos", "n_resources"] = df["num_threads"]
#     df.loc[df["parallelism_model"] == "omp", "n_resources"] = df["num_threads"]
#     df.loc[df["parallelism_model"] == "parlay", "n_resources"] = df["num_threads"]

#     # df.loc[df["parallelism_model"] == "mpi", "n_resources"] = df["num_procs"]
#     # df.loc[df["parallelism_model"] == "mpi+omp", "n_resources"] = df["num_procs"] * df["num_threads"]

#     # choose the row with min num_resources * runtime
#     df = df.groupby(["name", "parallelism_model", "output_idx"]).apply(
#             lambda row: row.iloc[np.argmin(row["runtime"] * row["n_resources"])]
#         ).reset_index(drop=True)

#     # use the min best_sequential_runtime
#     df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

#     # --- New Logic for Parlay and Other Models ---
#     # Isolate OMP runs to find the baseline for each problem
#     omp_baselines = df[df["parallelism_model"] == "omp"].copy()
#     omp_baselines["omp_baseline_runtime"] = omp_baselines.groupby("name")["best_sequential_runtime"].transform("min")
#     omp_baselines = omp_baselines[["name", "omp_baseline_runtime"]].drop_duplicates()

#     # Filter for Parlay runs and merge the OMP baseline
#     parlay_df = df[df["parallelism_model"] == "parlay"].copy()
#     parlay_df = parlay_df.merge(omp_baselines, on='name', how='left')
#     parlay_df.dropna(subset=['omp_baseline_runtime'], inplace=True)

#     # Calculate speedup_max for Parlay using the OMP baseline
#     parlay_results = parlay_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _efficiencyk(row["runtime"], np.min(row["omp_baseline_runtime"]), k, row["n_resources"], col_name='efficiency_max@{}')
#     ).reset_index()

#     # Filter for all other models
#     other_models_df = df[df["parallelism_model"] != "parlay"].copy()

#     # Calculate speedup_max for other models using their own sequential runtime
#     other_results = other_models_df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#         lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"], col_name='efficiency_max@{}')
#     ).reset_index()

#     # Combine the two results DataFrames
#     combined_results = pd.concat([parlay_results, other_results], ignore_index=True)

#     # compute the mean speedup_max@k
#     final_df = combined_results.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency_max@{k}": "mean"})


#     # # group by name, parallelism_model, and output_idx and call _efficiencyk
#     # df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
#     #         lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"], col_name='efficiency_max@{}')
#     #     ).reset_index()

#     # # compute the mean efficiency_max@k
#     # df = df.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency_max@{k}": "mean"})

#     return final_df

# def parse_problem_size(problem_size: str) -> int:
#     """ problem size is of format '(1<<n)' """
#     num = problem_size.split("<<")[1][:-1]
#     return 2 ** int(num)

# def main():
#     args = get_args()

#     # read in input
#     df = pd.read_csv(args.input_csv)

#     # read in problem sizes
#     with open(args.problem_sizes, "r") as f:
#         problem_sizes = json.load(f)
#         for problem in problem_sizes:
#             for parallelism_model, problem_size in problem_sizes[problem].items():
#                 df.loc[(df["name"] == problem) & (df["parallelism_model"] == parallelism_model), "problem_size"] = parse_problem_size(problem_size)

#     # remove rows where parallelism_model is kokkos and num_threads is 64
#     df = df[~((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 64))]

#     # filter/aggregate
#     df["did_run"] = df["did_run"].fillna(False)     # if it didn't build, then this will be nan; overwrite
#     df["is_valid"] = df["is_valid"].fillna(False)   # if it didn't build, then this will be nan; overwrite

#     # get only valid runs
#     valid_runs = get_correctness_df(df)
    
#     # get values for each k
#     all_results = []
#     # avg_ratios = []
#     for k in args.k:
#         build_values = buildk(df, k, args.problem_wise)
#         pass_values = passk(valid_runs, k, args.problem_wise)
#         speedup_values, avg_ratio = speedupk(df, k, args.n)
#         save_fastest_parlay_omp(df, k, args.n, f"fastest_parlayscheduler_omp_k{k}.json")
#         speedup_max_values = speedupk_max(df, k)
#         efficiency_values = efficiencyk(df, k, args.n)
#         efficiency_max_values = efficiencyk_max(df, k)
#         all_results.extend([build_values, pass_values, speedup_values, speedup_max_values, efficiency_values, efficiency_max_values])
#         # avg_ratios.append(avg_ratio)
    
#     # merge all_results; each df has one column and the same index
#     # build a new df with all the columns and the same index
#     merged_df = pd.concat(all_results, axis=1).reset_index()
#     # merged_ratios = pd.concat(avg_ratio, ignore_index=True)
#     merged_df = pd.merge(merged_df, avg_ratio, on="problem_type", how="left")


#     # if there were no successfull builds or runs, then speedup@k will be nan after merging
#     # replace NaN speedup@k values with 0.0
#     for k in args.k:
#         merged_df[f"speedup@{k}"] = merged_df[f"speedup@{k}"].fillna(0.0)
#         merged_df[f"speedup_max@{k}"] = merged_df[f"speedup_max@{k}"].fillna(0.0)
#         merged_df[f"efficiency@{k}"] = merged_df[f"efficiency@{k}"].fillna(0.0)
#         merged_df[f"efficiency_max@{k}"] = merged_df[f"efficiency_max@{k}"].fillna(0.0)

#     # add model name column
#     if args.model_name:
#         merged_df.insert(0, "model_name", args.model_name)

#     # clean up column names
#     column_name_map = {
#         "model_name": "model",
#         "parallelism_model": "execution model",
#         "problem_type": "problem type",
#     }
#     merged_df = merged_df.rename(columns=column_name_map)

#     # write to csv
#     if args.output:
#         merged_df.to_csv(args.output, index=False)
#     else:
#         pd.set_option('display.max_columns', merged_df.shape[1]+1)
#         pd.set_option('display.max_rows', merged_df.shape[0]+1)
#         print(merged_df)
        


# if __name__ == "__main__":
#     main()