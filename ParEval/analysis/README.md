# Analysis

This subdirectory contains scripts for analyzing the LLM outputs and driver
results.

`create-dataframe.py` -- convert results json files into CSV format

`metrics.py` -- compute pass@k, efficiency@k, speedup@k, and build@k for a 
particular results csv file

`metrics-scaling.py` -- compute the metrics at different resource counts; used
to get scaling results

`bin-the-stack.py` -- a utility script for analyzing The Stack dataset.

The arguments to each of these scripts can be found with `--help`. In general,
the workflow is to use `create-dataframe.py` to get a CSV results file for the
driver outputs and then feed this into `metrics.py` to get the relevant metrics.
This will in turn output another CSV file with the metrics divided by problem
type and execution model.

## `run-all.sh`
Automatically run `create-dataframe.py`, `metrics.py`, `compare-parlay-omp.py`, `obtain-fastest-impl.py`. 
We have a `run-all.sh` script that runs `create-dataframe.py`, `metrics.py` python scripts on all files in the `INPUT_DIR` and save them to `OUT_DIR`. You may modify `FASTEST_DIR` and `COMP_DIR`.


## `compare-parlay-omp.py`:
It takes the `*-out.csv` file and generate the numerical values for OMP code and Parlay code for each metric. The generated file is saved as `COMP_DIR/{model_name}.csv`

## `plot-parlay-omp-comp.py`:
Now, after running `compare-parlay-omp.py`, you could plot the parlay, omp values out across multiple models.

`--problem-wise` can be turned on to plot problem-wise histogram for each metric.

## `plot-metrics.py`:
This plots each metric against all problem types for all models in the input.