#!/bin/bash
#SBATCH -J bfs_original_eval
#SBATCH -p day
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=36
#SBATCH --mail-type=ALL
module load OpenMPI/5.0.3-GCC-13.3.0
g++ -O3 -std=c++17 -fopenmp bfs_original.cpp -I /home/ly337/project/gemini/parlaylib/include -o bfs_original
OMP_NUM_THREADS=1 PARLAY_NUM_THREADS=1 ./bfs_original 2000 >> bfs_eval_results/bfs_original_1.log
OMP_NUM_THREADS=1 PARLAY_NUM_THREADS=1 ./bfs_original 20000 >> bfs_eval_results/bfs_original_1.log
OMP_NUM_THREADS=1 PARLAY_NUM_THREADS=1 ./bfs_original 200000 >> bfs_eval_results/bfs_original_1.log
OMP_NUM_THREADS=1 PARLAY_NUM_THREADS=1 ./bfs_original 2000000 >> bfs_eval_results/bfs_original_1.log
OMP_NUM_THREADS=1 PARLAY_NUM_THREADS=1 ./bfs_original 20000000 >> bfs_eval_results/bfs_original_1.log
OMP_NUM_THREADS=2 PARLAY_NUM_THREADS=2 ./bfs_original 2000 >> bfs_eval_results/bfs_original_2.log
OMP_NUM_THREADS=2 PARLAY_NUM_THREADS=2 ./bfs_original 20000 >> bfs_eval_results/bfs_original_2.log
OMP_NUM_THREADS=2 PARLAY_NUM_THREADS=2 ./bfs_original 200000 >> bfs_eval_results/bfs_original_2.log
OMP_NUM_THREADS=2 PARLAY_NUM_THREADS=2 ./bfs_original 2000000 >> bfs_eval_results/bfs_original_2.log
OMP_NUM_THREADS=2 PARLAY_NUM_THREADS=2 ./bfs_original 20000000 >> bfs_eval_results/bfs_original_2.log
OMP_NUM_THREADS=4 PARLAY_NUM_THREADS=4 ./bfs_original 2000 >> bfs_eval_results/bfs_original_4.log
OMP_NUM_THREADS=4 PARLAY_NUM_THREADS=4 ./bfs_original 20000 >> bfs_eval_results/bfs_original_4.log
OMP_NUM_THREADS=4 PARLAY_NUM_THREADS=4 ./bfs_original 200000 >> bfs_eval_results/bfs_original_4.log
OMP_NUM_THREADS=4 PARLAY_NUM_THREADS=4 ./bfs_original 2000000 >> bfs_eval_results/bfs_original_4.log
OMP_NUM_THREADS=4 PARLAY_NUM_THREADS=4 ./bfs_original 20000000 >> bfs_eval_results/bfs_original_4.log
OMP_NUM_THREADS=8 PARLAY_NUM_THREADS=8 ./bfs_original 2000 >> bfs_eval_results/bfs_original_8.log
OMP_NUM_THREADS=8 PARLAY_NUM_THREADS=8 ./bfs_original 20000 >> bfs_eval_results/bfs_original_8.log
OMP_NUM_THREADS=8 PARLAY_NUM_THREADS=8 ./bfs_original 200000 >> bfs_eval_results/bfs_original_8.log
OMP_NUM_THREADS=8 PARLAY_NUM_THREADS=8 ./bfs_original 2000000 >> bfs_eval_results/bfs_original_8.log
OMP_NUM_THREADS=8 PARLAY_NUM_THREADS=8 ./bfs_original 20000000 >> bfs_eval_results/bfs_original_8.log
OMP_NUM_THREADS=16 PARLAY_NUM_THREADS=16 ./bfs_original 2000 >> bfs_eval_results/bfs_original_16.log
OMP_NUM_THREADS=16 PARLAY_NUM_THREADS=16 ./bfs_original 20000 >> bfs_eval_results/bfs_original_16.log
OMP_NUM_THREADS=16 PARLAY_NUM_THREADS=16 ./bfs_original 200000 >> bfs_eval_results/bfs_original_16.log
OMP_NUM_THREADS=16 PARLAY_NUM_THREADS=16 ./bfs_original 2000000 >> bfs_eval_results/bfs_original_16.log
OMP_NUM_THREADS=16 PARLAY_NUM_THREADS=16 ./bfs_original 20000000 >> bfs_eval_results/bfs_original_16.log
OMP_NUM_THREADS=32 PARLAY_NUM_THREADS=32 ./bfs_original 2000 >> bfs_eval_results/bfs_original_32.log
OMP_NUM_THREADS=32 PARLAY_NUM_THREADS=32 ./bfs_original 20000 >> bfs_eval_results/bfs_original_32.log
OMP_NUM_THREADS=32 PARLAY_NUM_THREADS=32 ./bfs_original 200000 >> bfs_eval_results/bfs_original_32.log
OMP_NUM_THREADS=32 PARLAY_NUM_THREADS=32 ./bfs_original 2000000 >> bfs_eval_results/bfs_original_32.log
OMP_NUM_THREADS=32 PARLAY_NUM_THREADS=32 ./bfs_original 20000000 >> bfs_eval_results/bfs_original_32.log
OMP_NUM_THREADS=36 PARLAY_NUM_THREADS=36 ./bfs_original 2000 >> bfs_eval_results/bfs_original_36.log
OMP_NUM_THREADS=36 PARLAY_NUM_THREADS=36 ./bfs_original 20000 >> bfs_eval_results/bfs_original_36.log
OMP_NUM_THREADS=36 PARLAY_NUM_THREADS=36 ./bfs_original 200000 >> bfs_eval_results/bfs_original_36.log
OMP_NUM_THREADS=36 PARLAY_NUM_THREADS=36 ./bfs_original 2000000 >> bfs_eval_results/bfs_original_36.log
OMP_NUM_THREADS=36 PARLAY_NUM_THREADS=36 ./bfs_original 20000000 >> bfs_eval_results/bfs_original_36.log
OMP_NUM_THREADS=64 PARLAY_NUM_THREADS=64 ./bfs_original 2000 >> bfs_eval_results/bfs_original_64.log
OMP_NUM_THREADS=64 PARLAY_NUM_THREADS=64 ./bfs_original 20000 >> bfs_eval_results/bfs_original_64.log
OMP_NUM_THREADS=64 PARLAY_NUM_THREADS=64 ./bfs_original 200000 >> bfs_eval_results/bfs_original_64.log
OMP_NUM_THREADS=64 PARLAY_NUM_THREADS=64 ./bfs_original 2000000 >> bfs_eval_results/bfs_original_64.log
OMP_NUM_THREADS=64 PARLAY_NUM_THREADS=64 ./bfs_original 20000000 >> bfs_eval_results/bfs_original_64.log
