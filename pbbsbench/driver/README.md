# Obtain Baseline Runtime + Generated Code Runtime
## Run baseline implementation to obtain runtime
```sh
python run-baseline-impl.py -i ../prompts/prompts.json --server XXXXXX
```
This generates the runtime for ParlayLib algorithm with both its internal scheduler and openmp scheduler on XXXXXX cluster with thread `1 2 4 8 16 32 36 48 64` on `testInputs` and `testInputs_small`.


## Gather baseline runtime 

```sh
python process-logfile.py --input ../generate/generate-gemini-2.5-pro.json --output_file baseline_runtime.json --log_dir_name baseline_runtime --log_file_format .log
```
If `TEST TERMINATED ABNORMALLY` appears in the log file, this log file and relevant information will be saved to `rerun-requests.json` automatically. You can use this information to rerun the tests. 


### Rerun abnormal ones:
```sh
chmod +x rerun-from-requests.py
./rerun-from-requests.py -i rerun-requests.json -b ../benchmarks -r 5
```
If you suspect dirty builds / leftover binaries, add:
```sh
./rerun-from-requests.py -i rerun-requests.json -b ../benchmarks -r 5 --clean-between
```


## Evaluate output generated 
Example:
```sh
python run-output.py --input ../generate/generate-gemini-2.5-test.json --server mac-air-m2-8GB --scheduler parlay --threads 1 2 4 8 16 32 36 48 64 --input-size small
```
## Rerun abnormal ones:
```sh
python run-output.py --rerun-input rerun-requests.json -s omp --server XXXXXX --input-size small
```
In rerun mode, it will use scheduler/server/input_size/threads from the JSON by default. If you want to override what’s in the rerun file (e.g., force threads):
```sh
python run-output.py --rerun-input rerun-requests.json \
  -s omp --force-scheduler \
  --server XXXXXX --force-server \
  --input-size large --force-input-size \
  -t 1 2 4 8 --force-threads
```


## Gather output statistics
```sh
python process-logfile.py --input ../generate/generate-gemini-2.5-pro.json --output_file processed-gemini-2.5-pro.json
```

```sh
python process-logfile.py --input ../generate/generate-gemini-2.5-test.json --output_file processed-gemini-2.5-test.json
```

## Plot Results

```sh
python plot-results.py -i processed-gemini-2.5-test.json -o gemini-2.5-pro-test

python plot-results.py -i processed_benchmark_data.json -o plots \
  --filter "algorithm=backForwardBFS,server=XXXXXX" --per_graph
```

This may be outdated:
```sh
python process-logfile.py --input ../generate/generate-gemini-2.5-pro.json --output_file baseline_runtime.json --log_dir_name eval_results_parallelDefs --log_file_format .log
```


## Flags as specified in `runall`:
```python
if (sys.argv.count("-h") > 0 or sys.argv.count("-help")):
    print("arguments:")
    print(" -force   : forces compile")
    print(" -nonuma  : do not use numactl -i all")
    print(" -scale   : run on a range of number of cores")
    print(" -par     : only run parallel benchmarks")
    print(" -notime  : only compile")
    print(" -nocheck : do not check results")
    print(" -small   : run on small data sets")
    print(" -keep    : keep temporary data files")
    print(" -ext     : extended set of benchmars")
    print(" -only <bnchmrk> : only run given benchmark")
    print(" -from <bnchmrk> : only run from given benchmark")
    forceCompile = True
    exit()
```


```sh
python new-run-output.py -i ../generate/outputs/prompts-with-baseline-impl.json -t 2 -s parlay --server XXXXXX --input-size small -m 4 --keep-logs
```