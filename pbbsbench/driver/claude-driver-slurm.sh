#!/bin/bash
#SBATCH -J claude_eval
#SBATCH -p day
#SBATCH -t 23:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
#SBATCH --mem=0
module load OpenMPI/4.1.4-GCC-12.2.0
module load jemalloc/5.3.0-GCCcore-12.2.0
module load GLib/2.75.0-GCCcore-12.2.0
module load Clang/13.0.1-GCCcore-12.2.0
module load numactl/2.0.16-GCCcore-12.2.0
module load Python/3.10.8-GCCcore-12.2.0

python run-output.py --input ../generate/generate-claude-opus-4-1.json --server XXXXXX --scheduler parlay --threads 1 2 4 8 16 32 36 48 64
python run-output.py --input ../generate/generate-claude-opus-4-1.json --server XXXXXX --scheduler omp --threads 1 2 4 8 16 32 36 48 64
