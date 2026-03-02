# Step-by-Step Training Script Setup for Code LLaMA-7B
1. Install dependencies
```sh
pip install transformers datasets accelerate bitsandbytes peft trl
```
2. Prepare the Dataset
Save your dataset in parlaylib_dataset.jsonl using this format:

Train using Trainer or SFTTrainer (from trl):
```sh
pip install trl
```


## OMP Data

- `omp_final_non_parallel.jsonl` contains `20453` serial cpp code. 
- `omp_final_parallel.jsonl` contains `38039` parallel cpp code. 
- `omp_all.jsonl` is the concatenated `jsonl` file. 

Source:
- `cpu_omp_unique.jsonl` is retrieved from [here](https://github.com/Scientific-Computing-Lab/MonoCoder/tree/main/data/OMP_Dataset/cpu/source).
- `omp_serial.jsonl` is retrieved from [here](https://github.com/LChenGit/OMP_serial_dataset)
- `cpp` code from `https://huggingface.co/datasets/HPC-Forran2Cpp/HPC_Fortran_CPP`