import argparse
import os 

def generate_duplicate(src, dst):
    if os.path.isfile(src):
        with open(src, "r") as f_src, open(dst, "w") as f_dst:
            f_dst.write(f_src.read())
        print(f"Created a copy: {dst}")
    else:
        print(f"Source file {src} does not exist.")

def generate_mix_check_files(filename, alg_dir):
    src = [f"{alg_dir}/../bench/{filename}Check.C",
           f"{alg_dir}/../bench/{filename}Time.C"]
    dst = [f"{alg_dir}/../bench/{filename}MixCheck.C",
           f"{alg_dir}/../bench/{filename}MixTime.C"]
    for s, d in zip(src, dst):
        if not os.path.isfile(d):
            generate_duplicate(s, d)
        else:
            print(f"File {d} already exists, skipping creation.")
    
def update_bnchmrk_to_mix(algorithm, alg_dir):
    testInputs_paths = [f"{alg_dir}/../bench/testInputs", 
                        f"{alg_dir}/../bench/testInputs_small"]
    for testInputs_path in testInputs_paths:
        if os.path.isfile(testInputs_path):
            with open(testInputs_path, "r") as f:
                lines = f.readlines()
            if testInputs_path.endswith("_small"):
                testInputs_path = testInputs_path.replace("_small", "Mix_small")
            else:
                testInputs_path = f"{testInputs_path}Mix"
            with open(testInputs_path, "w") as f:
                for line in lines:
                    if line.strip().startswith(f'bnchmrk="{algorithm}"'):
                        f.write(f'bnchmrk="{algorithm}Mix"\n')
                    elif line.strip().startswith(f'checkProgram="../bench/{algorithm}Check"'):
                        f.write(f'checkProgram="../bench/{algorithm}MixCheck"\n')
                    else:
                        f.write(line)
        else:
            print(f"File {testInputs_path} does not exist, skipping update.")
    

def generate_eval_scripts(filename, alg_dir, rounds, omp=False, server=''):
    thread_counts = [1, 2, 4, 8, 16, 32, 36, 48, 64, 70]
    # inputs = [2000, 20000, 200000, 2000000, 20000000]
    script_dir = f"{alg_dir}/eval_scripts"
    log_dir = f"{alg_dir}/eval_results"
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    if omp:
        script_name = os.path.join(script_dir, f"{filename}Mix_eval.sh")
        with open(script_name, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH -J {filename}_mix_eval
#SBATCH -p day
#SBATCH -t 3:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
module load OpenMPI/4.1.4-GCC-12.2.0
module load jemalloc/5.3.0-GCCcore-12.2.0
module load GLib/2.75.0-GCCcore-12.2.0
module load Python/3.10.8-GCCcore-12.2.0
module load Clang/13.0.1-GCCcore-12.2.0
module load numactl/2.0.16-GCCcore-12.2.0
make OPENMP=1
chmod +x testInputsMix
chmod +x testInputsMix_small

for p in {" ".join(map(str, thread_counts))}; do
  ./testInputsMix -r {rounds} -p $p > {log_dir}/{filename}Mix_{server}_${{p}}.log
done

for p in {" ".join(map(str, thread_counts))}; do
  ./testInputsMix_small -r {rounds} -p $p > {log_dir}/{filename}Mix_{server}_${{p}}_small.log
done

# make cleanall OPENMP=1
""")
    else:
        script_name = os.path.join(script_dir, f"{filename}_eval.sh")
        with open(script_name, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH -J {filename}_eval
#SBATCH -p day
#SBATCH -t 3:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
module load OpenMPI/4.1.4-GCC-12.2.0
module load jemalloc/5.3.0-GCCcore-12.2.0
module load GLib/2.75.0-GCCcore-12.2.0
module load Python/3.10.8-GCCcore-12.2.0
module load Clang/13.0.1-GCCcore-12.2.0
module load numactl/2.0.16-GCCcore-12.2.0
make


for p in {" ".join(map(str, thread_counts))}; do
  ./testInputs -r {rounds} -p $p > {log_dir}/{filename}_{server}_${{p}}.log
done

for p in {" ".join(map(str, thread_counts))}; do
  ./testInputs_small -r {rounds} -p $p > {log_dir}/{filename}_{server}_${{p}}_small.log
done

# make cleanall 
""")
            
    print(f"Scripts generated for threads: {thread_counts}\n")
    print(f"Script written to {script_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLURM scripts for BFS evaluation.")
    parser.add_argument("--filename", help="Base name of the .cpp and output executable (e.g., bfs_original or bfs_mixed)")
    parser.add_argument("--alg-dir", help="directory of algorithm")
    parser.add_argument("--rounds", type=int, help="Number of rounds for evaluation", default=5)
    parser.add_argument("--omp", action="store_true", help="compile with openmp")    
    parser.add_argument("--server", default='XXXXXX', help="Server name to append to log files")
    args = parser.parse_args()
    if args.omp:
        generate_mix_check_files(args.filename, args.alg_dir)
        update_bnchmrk_to_mix(args.filename, args.alg_dir)
    generate_eval_scripts(args.filename, args.alg_dir, args.rounds, args.omp, args.server)

# Note that it would be good if we compile with the original Makefile to obtain the original executables first
# then we can modify the Makefile to compile with OpenMP and obtain the mixed executables.
