from openevolve.controller import OpenEvolve
from openevolve.config import Config, LLMModelConfig
import csv
from pathlib import Path
import pandas as pd
import asyncio
import tempfile
import uuid
import os
import sys
import logging
from config import *


logging.getLogger('openevolve').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.INFO)

def find_latest_checkpoint(problem_result_dir):
    """Find the latest checkpoint in the problem directory"""
    checkpoint_base = problem_result_dir / "openevolve_output" / "checkpoints"
    if not checkpoint_base.exists():
        return None
    
    checkpoints = [d for d in checkpoint_base.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")]
    if not checkpoints:
        return None
    
    checkpoint_nums = []
    for cp in checkpoints:
        try:
            num = int(cp.name.split("_")[1])
            checkpoint_nums.append((num, cp))
        except:
            continue
    
    if not checkpoint_nums:
        return None
    
    latest = max(checkpoint_nums, key=lambda x: x[0])
    return str(latest[1])


def get_starting_iteration(checkpoint_path, results_file):
    """Get the starting iteration number from checkpoint or CSV"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load iteration from checkpoint metadata
        import json
        metadata_file = os.path.join(checkpoint_path, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata.get('last_iteration', 0)
        
        # Fallback: parse from checkpoint directory name
        checkpoint_name = os.path.basename(checkpoint_path)
        if checkpoint_name.startswith("checkpoint_"):
            try:
                return int(checkpoint_name.split("_")[1])
            except:
                pass
    
    # If no checkpoint, count from CSV (but this should rarely be used)
    if results_file.exists():
        import csv
        with open(results_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            last_iteration = 0
            for row in reader:
                if row and len(row) > 1:
                    try:
                        last_iteration = int(row[1])  # iteration column
                    except:
                        pass
            return last_iteration
    
    return 0


def load_problems_csv(csv_path='problems.csv'):
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if len(row) >= 2:
                problem_id = row[0]
                description = row[1] if len(row) > 1 else ''
                solution = row[2] if len(row) > 2 else ''
                data.append({
                    'problem_id': problem_id,
                    'description': description,
                    'solution': solution
                })
    return pd.DataFrame(data)


def get_description(problems_csv, problem_id):
    """Get full description of a problem"""
    problem = problems_csv[problems_csv['problem_id'] == problem_id]
    if not problem.empty:
        return problem.iloc[0]['description']
    return None


def prepare_evaluator(problem_id, problem_dir, starting_iteration, temp_dir, language):
    """Create a self-contained evaluator file"""
    
    # Get absolute path to the parent directory
    parent_dir = str(Path(__file__).parent.absolute())
    results_file = str(problem_dir.absolute() / "evaluation_results.csv")
    
    # Note: We escape the path string to handle Windows backslashes correctly if needed
    evaluator_code = f'''
import sys
import csv
from datetime import datetime
from pathlib import Path
import os
from openevolve.evaluation_result import EvaluationResult

# Add parent directory to path to import run_judge
sys.path.insert(0, r"{parent_dir}")
from run_judge import Judge

# Configuration
PROBLEM_ID = "{problem_id}"
RESULTS_FILE = Path(r"{results_file}")
ITERATION_COUNTER = [{starting_iteration}]

is_rust = "{language}" == "rust"

def evaluate(program_path):
    """Evaluate a program"""
    judge = Judge(PROBLEM_ID, time_limit=2, language="{language}")
    
    with open(program_path, 'r') as f:
        code = f.read()

    if is_rust:
        os.environ['CARGO_INCREMENTAL'] = '0'
        os.environ['CARGO_TERM_COLOR'] = 'never'
    
    # Strip Markdown
    code = code.strip()
    lines = code.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    code = chr(10).join(lines)
    
    judge_result = judge.evaluate(code, repeat=5)
    
    # Calculate combined_score
    if judge_result['passed']:
        if judge_result['avg_time'] > 0:
            combined_score = 1.0 / judge_result['avg_time']
        else:
            combined_score = 1000.0
    else:
        combined_score = 0.0

    # # --- CHANGED: Gradient Scoring Logic ---
    # combined_score = 0.0
    
    # # 1. Compilation Reward (Base Step)
    # # If there is no compile error, give a small foothold score
    # if not judge_result.get('compile_error'):
    #     combined_score += 1.0
        
    # # 2. Partial Test Credit (The Ladder)
    # # Award points proportional to tests passed (e.g., max 10 points for all tests)
    # total = judge_result.get('total_tests', 1)
    # passed = judge_result.get('passed_tests', 0)
    # if total > 0:
    #     pass_rate = passed / total
    #     combined_score += (pass_rate * 10.0)

    # # 3. Success Multiplier (The "Winner" Bonus)
    # # If it actually passes everything, boost the score significantly 
    # # and then add the efficiency metric.
    # if judge_result['passed']:
    #     combined_score += 100.0 # Big jump to separate valid solutions from partials
        
    #     # Add performance metric
    #     if judge_result.get('avg_time', 0) > 0:
    #         combined_score += (1.0 / judge_result['avg_time'])
    #     else:
    #         combined_score += 50.0 # fallback bonus
    
    # Increment iteration counter
    ITERATION_COUNTER[0] += 1
    
    # Append to CSV file incrementally
    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([
            datetime.now().isoformat(),
            ITERATION_COUNTER[0],
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
    
    # artifacts = judge_result.get('feedback', '') + judge_result.get('compile_error', '')

    feedback_msg = judge_result.get('feedback', '')
    compile_err = judge_result.get('compile_error', '')

    # If code compiles and passes almost all tests (e.g. > 90%), but fails the last ones
    # It is almost certainly a Time Limit Exceeded (TLE) or Memory Limit issue.
    # We explicitly tell the LLM to switch strategies.
    passed = judge_result['passed_tests']
    total = judge_result['total_tests']
    
    if not judge_result['passed'] and total > 0 and (passed / total) > 0.8:
        # Check max time to confirm TLE (assuming 2.0s is limit)
        if "Time Limit Exceeded." in feedback_msg:
            feedback_msg += "[SYSTEM WARNING]: CRITICAL FAILURE: TIME LIMIT EXCEEDED."
            feedback_msg += f"You passed {{passed}}/{{total}} tests, but the final tests timed out."
            feedback_msg += "Your complexity is likely O(N^2). You MUST rewrite the algorithm to be O(N log N) or O(N)."
            feedback_msg += "Do NOT check for edge cases. OPTIMIZE THE LOOPS."

    return EvaluationResult(
        metrics={{
            "combined_score": combined_score,
            "passed": judge_result['passed'],
            "tests_passed": judge_result['passed_tests'],
            "tests_total": judge_result['total_tests'],
            "avg_time": judge_result.get('avg_time', 0.0),
        }},
        artifacts={{
            "feedback": feedback_msg,
            "compile_error": judge_result.get('compile_error', '')
        }}
    )
'''
    
    # Write to temp file
    eval_file = os.path.join(temp_dir, f"evaluator_{uuid.uuid4().hex[:8]}.py")
    with open(eval_file, "w") as f:
        f.write(evaluator_code)
    
    return eval_file


async def run_evolution_async(problem_id, problem_dir, problem_description, checkpoint_path=None, language="cpp"):
    """Run evolution using OpenEvolve class directly"""
    
    results_file = problem_dir / "evaluation_results.csv"
    
    # Initialize CSV file with headers if starting fresh
    if not results_file.exists():
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                'timestamp', 'iteration', 'combined_score', 'passed', 
                'tests_passed', 'tests_total', 'avg_time', 'variance_time', 'max_time', 
                'total_time', 'compile_error', 'code'
            ])
    
    # Count existing evaluations to continue iteration numbering
    starting_iteration = get_starting_iteration(checkpoint_path, results_file)
    
    # Create temp directory for evaluator file
    temp_dir = tempfile.mkdtemp(prefix="openevolve_eval_")
    
    try:
        # Create self-contained evaluator file
        evaluator_file = prepare_evaluator(problem_id, problem_dir, starting_iteration, temp_dir, language)
        
        # Create initial C++ program (in code, not file)
        if language == "cpp":
            initial_program = """//EVOLVE-BLOCK-START
#include <iostream>
#include <parlay/sequence.h>
#include <parlay/primitives.h>
using namespace std;

int main() {
    // TODO: Implement solution to the problem
    // Read input, process it, and output the result
    return 0;
}
//EVOLVE-BLOCK-END
"""
            # Write initial program to temp file for OpenEvolve
            initial_program_file = problem_dir / "initial_program.cpp"

            with open(initial_program_file, 'w') as f:
                f.write(initial_program)
        
            # Create system prompt
            system_prompt = f"""You are an expert C++ competitive programmer. Your task is to write a COMPLETE, CORRECT, and FAST C++ solution.

PROBLEM:
{problem_description}

REQUIREMENTS:
1. Write a complete C++ parallel program that compiles and runs correctly
2. Read input from standard input (cin)
3. Write output to standard output (cout)
4. Handle all edge cases mentioned in the problem
5. Optimize for speed - use efficient algorithms and data structures
6. Use C++ STL where appropriate (vector, map, set, priority_queue, etc.)
7. Consider time complexity and space complexity
8. The parlay library MUST be used as the core computation of the program

AVAILABLE LIBRARIES:
- Standard C++ libraries (iostream, algorithm, vector, map, etc.)
- The parlay library

Note:
- `parlay::parallel_for` does not guarantee ordering, do not use it with IO operations.

CODE STYLE:
- Use C++ style comments: // for single line, /* */ for multi-line
- Do NOT use Python-style # comments
- Comments should be simple and short
- Include necessary headers
- Write clean, readable code

OUTPUT FORMAT:
Return ONLY the complete C++ code. Do not include explanations, markdown formatting, or code blocks.
Just the raw C++ source code that can be directly compiled.
"""
        elif language == "rust":
            initial_program = """//EVOLVE-BLOCK-START
use std::io::{self, Read};
use rayon::prelude::*;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    // TODO: parse input, compute, print output
}
//EVOLVE-BLOCK-END
"""
            initial_program_file = problem_dir / "initial_program.rs"
            with open(initial_program_file, 'w') as f:
                f.write(initial_program)
        
            system_prompt = f"""You are an expert Rust programmer. Solve the following problem.

CRITICAL INSTRUCTIONS:
1. OUTPUT CODE ONLY. Do not output any thinking or planning in comments.
2. NO 'Chain of Thought' reasoning.
3. The response must be valid Rust code that compiles immediately.

Problem:
{problem_description}

ENVIRONMENT:
- Rust edition 2021
- Allowed crates: Rust standard library + rayon (rayon::prelude::*)
- You are allowed to use other crates or unsafe code.

ABSOLUTE OUTPUT RULE:
Return ONLY the complete Rust source code that can be directly compile. No explanations. No markdown. No code fences. No extra text.

CORE GOAL:
Make the solution pass all tests. Hidden tests include tricky edge cases and worst-case constraints.

REQUIREMENTS:
- Read input from standard input (cin)
- Write output to standard output (cout)
- Handle all edge cases mentioned in the problem
- Optimize for speed - use efficient algorithms and data structures
- Include #[derive(Clone)] and other necessary traits where appropriate
- Use Rayon where appropriate 
- Consider time complexity and space complexity
- The rayon library MUST be used as the core computation of the program
- No debug prints.
- Deterministic output.

CODE STYLE:
- Use Rust style comments: // for single line, /* */ for multi-line
- Do NOT use Python-style # comments
- Comments should be simple and short. **Do not include EXCESSIVE comments!!!**
- Include necessary headers
- Write clean, readable code
"""

        
        # Create config
        config = Config()
        config.max_iterations = 30
        config.checkpoint_interval = 5
        
        config.max_code_length = 100000
        
        # Configure LLM
        # NOTE: OpenEvolve uses litellm under the hood usually, so we pass
        # the model name and API base. 
        # If Qwen is hosted on vLLM/Ollama, it is OpenAI-compatible.
        config.llm.models = [
            LLMModelConfig(
                name=f"{MODEL_NAME}", 
                api_key=API_KEY,
                api_base=API_BASE,
                weight=0.1,
                temperature=1,
                top_p=0.9,
                max_tokens=os.getenv("CCD_LLM_MAX_TOKEN", 65536), # Adjusted for typical 30B model context limits
                timeout=600
            ),
        ]
        
        # Configure database
        config.database.population_size = 25
        config.database.num_islands = 1
        
        # Configure evaluator
        config.evaluator.timeout = 600
        config.evaluator.cascade_evaluation = False
        config.diff_based_evolution = False
        config.allow_full_rewrites = True
        
        # Configure prompt
        config.prompt.system_message = system_prompt
        config.prompt.num_top_programs = 3
        config.prompt.num_diverse_programs = 10
        
        config.evaluator.enable_artifacts = True
        config.prompt.include_artifacts = True

        print(dir(config.prompt))
        
        # Create OpenEvolve controller
        controller = OpenEvolve(
            initial_program_path=str(initial_program_file),
            evaluation_file=evaluator_file,
            config=config,
            output_dir=str(problem_dir / "openevolve_output")
        )

        
        # Run evolution with checkpoint support
        print(f"Starting evolution for problem: {problem_id}")
        print(f"Model: {MODEL_NAME} @ {API_BASE}")
        print(f"Results will be saved to: {problem_dir}")
        print(f"CSV log: {results_file}")
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            print(f"Starting from iteration: {starting_iteration}")
        
        best_program = await controller.run(
            checkpoint_path=checkpoint_path
        )
        
        # Save best solution
        if best_program:
            ext = "cpp" if language == "cpp" else "rs"
            with open(problem_dir / f"best_solution.{ext}", "w") as f:
                f.write(best_program.code)
            
            # Save summary
            summary_file = problem_dir / "summary.csv"
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value'])
                writer.writerow(['problem_id', problem_id])
                writer.writerow(['best_score', best_program.metrics.get('combined_score', 0.0)])
                writer.writerow(['has_solution', True])
                for key, value in best_program.metrics.items():
                    writer.writerow([f'metric_{key}', value])
            
            print(f"\n=== Evolution Complete ===")
            print(f"Best Score: {best_program.metrics.get('combined_score', 0.0)}")
            print(f"Best solution saved to: {problem_dir / 'best_solution.cpp'}")
            print(f"Summary: {summary_file}")
        else:
            print("No solution found")
    
    finally:
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def main():
    problems_csv = load_problems_csv()
    if len(sys.argv) < 2:
        print("Usage: python run_openevolve.py <problem_id>")
        sys.exit(1)
    
    problem_id = sys.argv[1]
    
    # Create problem-specific directory
    problem_dir = RESULTS_DIR / problem_id
    problem_dir.mkdir(parents=True, exist_ok=True)
    
    problem_description = get_description(problems_csv, problem_id)
    if problem_description is None:
        raise ValueError(f"Problem description for {problem_id} is not found")
    
    # Save problem description
    with open(problem_dir / "problem_description.txt", 'w') as f:
        f.write(problem_description)
    
    # Find checkpoint if exists
    checkpoint_path = find_latest_checkpoint(problem_dir)
    
    # Run evolution
    try:
        asyncio.run(run_evolution_async(problem_id, problem_dir, problem_description, checkpoint_path, LANGUAGE))
    except Exception as e:
        print(f"Error during evolution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()