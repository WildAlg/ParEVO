import argparse
import os 

def generate_bfs_scripts(filename, omp=False):
    thread_counts = [1, 2, 4, 8, 16, 32, 36, 64]
    cpp_file = f"{filename}.cpp"
    exe_file = filename
    inputs = [2000, 20000, 200000, 2000000, 20000000]
    parlay_include = "/home/ly337/project/gemini/parlaylib/include"
    script_dir = f"bfs_eval_scripts"
    log_dir = f"bfs_eval_results"
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    script_name = os.path.join(script_dir, f"{filename}_eval.sh")
    with open(script_name, "w") as f:
        if omp:
            f.write(f"""#!/bin/bash
#SBATCH -J {filename}_eval
#SBATCH -p day
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=36
#SBATCH --mail-type=ALL
module load OpenMPI/5.0.3-GCC-13.3.0
g++ -O3 -std=c++17 -fopenmp {cpp_file} -I {parlay_include} -o {exe_file}
""")
            for thread in thread_counts:
                log_file = os.path.join(log_dir, f"{filename}_{thread}.log")
                for inp in inputs:
                    f.write(f"OMP_NUM_THREADS={thread} PARLAY_NUM_THREADS={thread} ./{exe_file} {inp} >> {log_file}\n")
                
                
        else:
            f.write(f"""#!/bin/bash
#SBATCH -J {filename}_eval{thread}
#SBATCH -p day
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=36
#SBATCH --mail-type=ALL
module load OpenMPI/5.0.3-GCC-13.3.0
g++ -O3 -std=c++17 -I {parlay_include} {cpp_file} -o {exe_file}
""")
            for thread in thread_counts:
                log_file = os.path.join(log_dir, f"{filename}_{thread}.log")
                for inp in inputs:
                    f.write(f"PARLAY_NUM_THREADS={thread} ./{exe_file} {inp} >> {log_file}\n")
                
            
    print(f"Scripts generated for threads: {thread_counts}\ninputs: {inputs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLURM scripts for BFS evaluation.")
    parser.add_argument("filename", help="Base name of the .cpp and output executable (e.g., bfs_original or bfs_mixed)")
    parser.add_argument("omp", help="compile with openmp", default=False)    
    args = parser.parse_args()
    generate_bfs_scripts(args.filename, args.omp)
