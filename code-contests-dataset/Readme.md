Tested on WSL: Ubuntu-24.04.

## Prerequisite
`pip install dmoj`
(If the install fail on linux:
`sudo apt install libseccomp-dev`)

```
git clone --recursive https://github.com/DMOJ/judge-server.git
cd judge-server
pip install -e .
```

Get the dmoj problems zip and unzip it under [./dmoj_problems](./dmoj_problems). Run `python prepare.py` to unzip the test cases in the [./dmoj_problems](./dmoj_problems).


## Main Files
- `judge.yml`: dmoj judger required environment setup
- `problem_classify.py`: Use a LLM to classify graph / parallel problems
    - `results.csv`: Records of the classifier's answers
    - `interested_problems.csv`: from results.csv but only ones with "YES"
- `problem_preview.py`: Some helper functions to retrieve description of the problems
- `problems.csv`: A list of [id, description, solution] of dmoj problems
- `run_judge.py`: A coustom judger that given source code, can compile the program and run test cases with statistics
- `run_openevolve.py`: A workflow that runs openevolve on a problem with customizable configs
- `viewer.py`: Hosts a website that previews the result (codes and scores) from `run_openevolve.py`, usage `python viewer.py {problem_id}`

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
Use run_openevolve.py with suitable parameters