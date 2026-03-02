#!/bin/bash
#SBATCH -J baseline_runtime
#SBATCH -p day
#SBATCH -t 23:30:00
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
#SBATCH --mem=0

# Enable unlimited core dump file size
ulimit -c unlimited

module load OpenMPI/4.1.4-GCC-12.2.0
module load jemalloc/5.3.0-GCCcore-12.2.0
module load GLib/2.75.0-GCCcore-12.2.0
module load Clang/13.0.1-GCCcore-12.2.0
module load numactl/2.0.16-GCCcore-12.2.0
module load Python/3.10.8-GCCcore-12.2.0

python run-baseline-impl.py -i ../prompts/prompts-test.json --server XXXXXX
# python process-logfile.py --input ../generate/generate-gemini-2.5-pro.json --output_file baseline_runtime.json --log_dir_name baseline_runtime --log_file_format .log 
# python process-logfile.py --input ../generate/generate-gemini-2.5-pro.json --output_file claude-output.json --log_dir_name eval-output --log_file_format .txt 
