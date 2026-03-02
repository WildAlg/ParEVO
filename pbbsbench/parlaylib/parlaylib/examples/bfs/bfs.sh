#!/bin/bash
#SBATCH -J bfs
#SBATCH -p day
#SBATCH -t 5:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
module load OpenMPI/4.1.4-GCC-12.2.0

g++ -O3 -std=c++17 -fopenmp bfs_mixed.cpp -I /home/ly337/project_pi_ql324/ly337/gemini/parlaylib/include -o bfs_mixed

for p in 1 2 4 8 16 32 36 64 70; do
    for size in 1000000 10000000 100000000; do
        PARLAY_NUM_THREADS=$(p) ./bfs_mixed $(size) >> eval_results/bfs_mixed_$(p)_$(size).log
    done
done


# make cleanall