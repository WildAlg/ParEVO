**Note: most of the functionalities requires problem testcases (available upon request).**

## Prerequisite

`pip install dmoj`
(If the install fail on linux: try `sudo apt install libseccomp-dev` first)

```
git clone --recursive https://github.com/DMOJ/judge-server.git
cd judge-server
pip install -e .
```

Get the dmoj problem testcases zip (available upon request) and unzip it under [./dmoj_problems](./dmoj_problems). Run `python prepare.py` to unzip the test cases in the [./dmoj_problems](./dmoj_problems).


## Main Files
- `judge.yml`: Dmoj judger required environment setup
- `config.py`: Configurations shared among all scripts
- `problem_classify.py`: Use a LLM to classify graph / parallel problems
- `problems.csv`: A list of [id, description, solution] of dmoj problems
- `run_judge.py`: A custom judger that given source code, can compile the program and run test cases with statistics
- `run_openevolve.py`: A workflow that runs openevolve on a problem with config specified in `config.py`
- `viewer.py`: Hosts or generate html of a website that previews the result (codes and scores) from `run_openevolve.py`, usage `python viewer.py {problem_id}` or `python viewer.py --export {problem_id}`
- `dmoj_training_problems/{problem_id}/`: The ECA running results of `{problem_id}` with gemini 3 pro.
    - baseline.cpp: The first valid solution
    - best_iter_x.cpp: Best valid solution over first x iterations
    - evaluation_results.csv: A csv log of all iterations scores, runtimes, errors, and codes

## Slurm Modules
```sh
module load Rust/1.83.0-GCCcore-13.3.0
module load OpenSSL/3 
module load miniconda
module load GCC/13.3.0      
module load GCCcore/13.3.0 

conda activate huggingface
```

## Visualizer
See viewer.py

## Openevolve
Use `run_openevolve.py` with suitable parameters in `config.py`