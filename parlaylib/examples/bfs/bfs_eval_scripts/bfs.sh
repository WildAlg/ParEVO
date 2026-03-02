#!/bin/bash
#SBATCH -J bfs_parlay_eval1
#SBATCH -p day
#SBATCH -t 3:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL

module load OpenMPI/5.0.3-GCC-13.3.0

g++ -O3 -pthread -std=c++17 -I /home/ly337/project_pi_ql324/ly337/gemini/parlaylib/include bfs_original.cpp -o bfs_original

for p in 1 2 4 8 16 32 36 64 70; do
    for size in 1000000 10000000 100000000; do
        PARLAY_NUM_THREADS=${p} ./bfs_original ${size} >> eval_results/bfs_original_${p}_${size}.log
    done
done
