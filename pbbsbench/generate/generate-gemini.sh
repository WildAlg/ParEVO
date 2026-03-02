#!/bin/bash
#SBATCH -J gemini-generate
#SBATCH -p day
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --mem=0
module load miniconda
conda activate huggingface

api="XXXXXX"
python generate-gemini.py --model gemini-2.5-pro --prompts ../prompts/pbbs-generation-prompts-test.json --output generate-gemini-2.5-pro-test.json --num-samples-per-prompt 1 --api-key ${api} --max-new-tokens 20000 
python process-output.py -i generate-gemini-2.5-pro.json