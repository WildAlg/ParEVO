#!/bin/bash
#SBATCH -J gpt5-generate
#SBATCH -p day
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --mem=0
module load miniconda
conda activate huggingface

python generate-openai.py --model gpt-5 --effort high --prompts ../prompts/prompts.json --output generate-gpt5.json --num-samples-per-prompt 3 --api-key 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 
python process-output.py -i generate-gpt5.json 