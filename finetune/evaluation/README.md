
## Evaluate Model Generation Capability

### `test_code_hollow` attribute
Generate a new attribute `test_code_hollow`. This should be used as part of the code completion prompt. The generated code should be compared against the full `test_code`. 
```sh
python helpers.py rm_llm data_test.jsonl -o data_test_hollow.jsonl
```

### `rm_assert`
Only the full `test_code` of the `data_train.jsonl` should be removed of their assertions, because they potentially serve as training data. 
```sh
python helpers.py rm_assert data_train.jsonl -o data_train_noassert.jsonl
```

### Generate metric scores 
```sh
python evaluation.py metric ../generated-examples/data_test_hollow.jsonl codellama/CodeLlama-7b-hf --save_metric_scores
```
This saves the output of `codellama/CodeLlama-7b-hf` to `eval_output.jsonl` by default. If `--save_metric_scores` is turned on, the generated metric scores (CodeBLEU, BLEU, ROUGE-L, Chrf) are saved at `metric_scores.txt`. The input prompt for the inference is `{instruction}\n{input}\n{test_code_hollow}`. Depending on the number of parameters of the model, user should choose GPUs with appropriate vRAM. 

**Models**:
```sh
deepseek-ai/deepseek-coder-6.7b-base
deepseek-ai/deepseek-coder-7b-base-v1.5
codellama/CodeLlama-7b-hf
Qwen/Qwen3-8B
codellama/CodeLlama-13b-hf
Phind/Phind-CodeLlama-34B-v2
```
