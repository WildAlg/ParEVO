# "codellama/CodeLlama-7b-hf"
# List of model paths to iterate through
MODELS=(
    "deepseek-ai/deepseek-coder-6.7b-base"
    "deepseek-ai/deepseek-coder-6.7b-instruct"
    "deepseek-ai/deepseek-coder-7b-base-v1.5"
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    "codellama/CodeLlama-7b-hf"
    "codellama/CodeLlama-7b-Instruct-hf"
    "Qwen/Qwen3-8B"
    "codellama/CodeLlama-13b-hf"
    "Phind/Phind-CodeLlama-34B-v2"
    "hpcgroup/hpc-coder-v2-6.7b"
)

# Define the common test file path
testfile="../generated-examples/data_test_hollow.jsonl"

# Loop through each model in the array
for model in "${MODELS[@]}"; do
    echo "Starting evaluation for model: $model"

    # --- Per-Model Cache Setup ---
    # Use a unique cache directory for each model run to ensure isolation.
    export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
    export TRANSFORMERS_CACHE=$HF_HOME
    rm -rf "$HF_HOME" # Ensure the directory is clean from any previous runs
    mkdir -p "$HF_HOME"
    echo "Using HF cache at $HF_HOME"
    # --- End Per-Model Cache Setup ---

    # Sanitize the model name to create a valid filename for output.
    # This replaces slashes and special characters with hyphens.
    sanitized_model_name=$(echo "$model" | tr -c '[:alnum:]' '-')
    output="test_output/eval-output-${sanitized_model_name}.jsonl"
    output_complete="test_output/eval-output-complete-${sanitized_model_name}.jsonl"

    # Run the evaluation script with the current model and dynamic output file
    python evaluation.py metric "${testfile}" "${model}" --save_metric_scores --output_file "${output}"
    python evaluation.py metric "${testfile}" "${model}" --save_metric_scores --output_file "${output_complete}" --prompt_complete_code

    # Check the exit status of the python command
    if [ $? -eq 0 ]; then
        echo "Successfully completed evaluation for $model"
    else
        echo "Evaluation failed for $model"
    fi

    # --- Per-Model Cache Cleanup ---
    # Remove the cache directory immediately after the evaluation for the current model.
    rm -rf "$HF_HOME"
    echo "Removed HF cache at $HF_HOME"
    # --- End Per-Model Cache Cleanup ---

    echo "---" # Separator for readability in logs
done

echo "All evaluations finished at $(date)"
