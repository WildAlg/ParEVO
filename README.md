# ParEVO

A system that integrates human expert context with an evolutionary LLM agent.

## Paper and Website

- **Project Website**: [https://quanquancliu.com/ParEVO/index.html](https://quanquancliu.com/ParEVO/index.html)
- **Paper**: [ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution](https://quanquancliu.com/ParEVO/main.pdf)

## Citation

If you use ParEVO or the Parlay-Instruct Corpus in your research, please use the following citation:

```bibtex
@inproceedings{yang2026parevo,
  title={ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution},
  author={Yang, Liu and Nie, Zeyu and Liu, Andrew and Zou, Felix and Altinb{\"u}ken, Deniz and Yazdanbakhsh, Amir and Liu, Quanquan C.},
  booktitle={ICML},
  year={2026}
}
```

## PBBS Benchmarks
We use the PBBS Benchmarks (https://cmuparlay.github.io/pbbsbench/), where for each benchmark the suite provides:
* the specification of the input and expected output for the problem,
* the specification of a default set of input instances,
* code for generating inputs (written to a file),
* code for checking correctness of output (read from a file),
* code for timing the benchmark across the instances,
* a default parallel implementation,
* a default sequential implementation (for most benchmarks),
* a variety of other implementations (for some benchmarks).

### Set up Environment 
Create a Conda environment and install packages.
```sh
conda create -n gemini-env python=3.12
conda install pytorch torchvision torchaudio -c pytorch
conda install transformers
conda install -c conda-forge sentencepiece
```

### Example: deterministicBFS
1. Navigate to `pbbsbench/benchmarks/breadthFirstSearch/deterministicBFS`, run `make` to generate the executable of the driver code. Note that the driver code are the `BFSCheck.C` and `BFSTime.C` in `pbbsbench/benchmarks/breadthFirstSearch/bench`. If you need openmp, run with `make OPENMP=1`.
2. Test it with command line in the form of 
```sh
<bench> [-o <outfile>] [-r <numrounds>] [-scale <num_of_thread>] <infile>
```
3. Check the correctness: navigate to `pbbsbench/benchmarks/breadthFirstSearch/bench`, run 
```sh
<checker> <input> <output>
```
4. Within each implementation directory, you can run also run `./testInputs`. On a machine with multiple chips, using `umactl -i all ./testInputs` will give better results. `./testInputs_small` will use the smaller inputs. The testInputs script has several options including:

```sh
-x : do not check the output
-r <count>  : number of rounds to use
-p <count>  : number of threads to use
```
The actual inputs are specified in the script and can be changed if desired. So we can probably change our implementation of BFS, `make` and run `./testInputs -r <count> -p <count> >> <output_log>` to write the evaluation results to the `output_log`. An example of the output:
```sh
randLocalGraph_J_10_20000000 :  -r 3 -o /tmp/ofile10284_857347 : '2.257', '2.255', '2.257', geomean = 2.256
rMatGraph_J_12_16000000 :  -r 3 -o /tmp/ofile859821_410881 : '2.476', '2.462', '2.467', geomean = 2.468
```

## ParEval
### Create conda environment
```sh
conda create --name huggingface python=3.11.* transformers accelerate tokenizers datasets jupyter jupyterlab
```

### [prompts](ParEval/prompts)
We added `parlay` and `cilk` to the raw prompts by `create-parlay-raw-prompts.py` (which leverages gemini API to translate the OpenMP prompt and function definition to that of Parlaylib) and `update-prompts-concise.py` (which traverses the `raw` directory and replaces each `OpenMP` with `OpenCilk`).  Then, we can run `gather-raw-prompts.py` to genterate `generation-prompts.json`. 

### [generate](ParEval/generate)
- **Gemini**: Use `generate-gemini.py`
- **Local Models with vLLM**: Use `generate-vllm.py`
