#!/bin/bash
#SBATCH -J claude-generate
#SBATCH -p day
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --mem=0
module load miniconda
conda activate huggingface

api="XXXXXX"
python generate-claude.py --model claude-opus-4-1 --prompts ../prompts/prompts.json -o generate-claude-opus-4-1.json --num-samples-per-prompt 3 --api-key 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
python process-output.py -i generate-claude-opus-4-1.json 