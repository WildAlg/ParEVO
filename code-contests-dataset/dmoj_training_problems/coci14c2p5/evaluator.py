
import csv
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import Judge
sys.path.insert(0, r"/home/vitalemontea/work/gemini")
from run_judge import Judge

problem_id = "coci14c2p5"
results_file = Path(r"/home/vitalemontea/work/gemini/code_contest_dataset/results/coci14c2p5/evaluation_results.csv")
iteration_counter = [0]

judge = Judge(problem_id, time_limit=2)

def evaluate(program_path):
    """Evaluate a C++ program"""
    global iteration_counter
    
    with open(program_path, 'r') as f:
        code = f.read()
    
    judge_result = judge.evaluate(code, repeat=5)
    
    # Calculate combined_score
    if judge_result['passed']:
        if judge_result['avg_time'] > 0:
            combined_score = 1.0 / judge_result['avg_time']
        else:
            combined_score = 1000.0
    else:
        combined_score = 0.0
    
    if not judge_result['passed'] and 'compile_error' in judge_result:
        judge_result['artifacts'] = {
            'compile_error': judge_result['compile_error'],
            'stderr': judge_result.get('compile_error', '')
        }
    
    # Increment iteration counter
    iteration_counter[0] += 1
    
    # Append to CSV file incrementally
    with open(results_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([
            datetime.now().isoformat(),
            iteration_counter[0],
            combined_score,
            judge_result['passed'],
            judge_result['passed_tests'],
            judge_result['total_tests'],
            judge_result.get('avg_time', 0.0),
            judge_result.get('variance_time', 0.0),
            judge_result.get('max_time', 0.0),
            judge_result.get('total_time', 0.0),
            judge_result.get('compile_error', ''),
            code
        ])
    
    return {
        "combined_score": combined_score,
        "passed": judge_result['passed'],
        "tests_passed": judge_result['passed_tests'],
        "tests_total": judge_result['total_tests'],
        "avg_time": judge_result.get('avg_time', 0.0),
        "artifacts": judge_result.get('artifacts', "")
    }
