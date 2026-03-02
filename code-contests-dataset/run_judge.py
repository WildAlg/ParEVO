import yaml
import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import resource
import time
from typing import List, Dict, Any
from dmoj import judgeenv
from dmoj.cptbox.filesystem_policies import RecursiveDir
from dmoj.result import CheckerResult
import glob
from config import *
import os

os.environ["RAYON_NUM_THREADS"] = "64"
# print("RUN_JUDGE_PATH:", __file__)
if 'LD_LIBRARY_PATH' in os.environ:
    # Print it to verify it's being seen
    print(f"DEBUG: LD_LIBRARY_PATH is set to: {os.environ['LD_LIBRARY_PATH']}")
else:
    print("WARNING: LD_LIBRARY_PATH is not set. OpenSSL might fail.")
# --------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
parlaylib_include_path = os.path.join(ROOT_DIR, "parlaylib", "include")

OPENSSL_LIB_PATH = "/apps/software/2024a/software/OpenSSL/3/lib64"
OPENSSL_INCLUDE_PATH = "/apps/software/2024a/software/OpenSSL/3/include"
RUST_LIB_PATH    = "/apps/software/2024a/software/Rust/1.83.0-GCCcore-13.3.0/lib"
GCC_ROOT_PATH     = "/apps/software/2024a/software/GCCcore/13.3.0"
GCC_BIN_DIR      = os.path.join(GCC_ROOT_PATH, "bin")
GCC_BIN  = os.path.join(GCC_ROOT_PATH, "bin", "g++")
GCC_C_BIN        = os.path.join(GCC_BIN_DIR, "gcc")
ALL_SOFTWARE_PATH = "/apps/software/2024a/software"

# --- FIX: Force System PATH to prioritize GCC 13 ---
# This ensures that any subprocess calling 'g++' finds the new one,
# even if DMOJ bypasses our python patches.
os.environ["PATH"] = f"{GCC_BIN_DIR}:{os.environ['PATH']}"
os.environ["CC"] = GCC_C_BIN
os.environ["CXX"] = GCC_BIN

current_ld = os.environ.get('LD_LIBRARY_PATH', '')
# Prepend OpenSSL path
if OPENSSL_LIB_PATH not in current_ld:
    os.environ['LD_LIBRARY_PATH'] = f"{OPENSSL_LIB_PATH}:{current_ld}"
    print(f"DEBUG: Added OpenSSL to LD_LIBRARY_PATH: {OPENSSL_LIB_PATH}")

def ensure_rayon_built():
    """
    Checks if Rayon is built. If not, creates a temporary Cargo project
    and builds it to generate the .rlib file.
    """
    # We will build in a dedicated temp folder to keep things clean
    build_dir = os.path.join(ROOT_DIR, "temp_rayon_build")
    target_dir = os.path.join(build_dir, "target")
    deps_dir = os.path.join(target_dir, "release", "deps")
    
    # Check if a compiled Rayon library already exists
    # We use glob because the hash in the filename (librayon-<hash>.rlib) changes
    if os.path.exists(deps_dir):
        existing_libs = glob.glob(os.path.join(deps_dir, "librayon-*.rlib"))
        if existing_libs:
            # Found it! Return the folder and the first match
            return deps_dir, existing_libs[0]

    print("⚠️  Rayon library not found. Building dependency... (this runs once)")
    
    # 1. Create the build directory
    os.makedirs(os.path.join(build_dir, "src"), exist_ok=True)
    
    # 2. Create a minimal Cargo.toml
    with open(os.path.join(build_dir, "Cargo.toml"), "w") as f:
        f.write("""
[package]
name = "build_deps"
version = "0.1.0"
edition = "2021"

[dependencies]
rayon = "1.8"
""")
        
    # 3. Create a dummy main.rs (needed for cargo build)
    with open(os.path.join(build_dir, "src", "main.rs"), "w") as f:
        f.write("fn main() {}")
        
    # 4. Run cargo build --release
    try:
        subprocess.run(
            ["cargo", "build", "--release"], 
            cwd=build_dir, 
            check=True,
            stdout=subprocess.DEVNULL, # Hide build output unless it fails
            stderr=subprocess.PIPE     # Capture stderr in case of error
        )
    except subprocess.CalledProcessError as e:
        print("Error building Rayon:")
        print(e.stderr.decode())
        raise RuntimeError("Failed to build Rayon dependency.")
    
    # 5. Find the generated library
    libs = glob.glob(os.path.join(deps_dir, "librayon-*.rlib"))
    if not libs:
        raise RuntimeError("Build finished but no librayon-*.rlib found.")
    
    return deps_dir, libs[0]

if LANGUAGE == "rust":
    # Initialize dependencies dynamically
    host_deps_folder, SANDBOX_RAYON_FILE = ensure_rayon_built()
    SANDBOX_FOLDER = host_deps_folder

    print(f"DEBUG: Using Rayon lib: {SANDBOX_RAYON_FILE}")


# Load config
with open(os.path.join(ROOT_DIR, 'judge.yml'), 'r') as f:
    config = yaml.safe_load(f)
    judgeenv.env.update(config)

judgeenv.skip_self_test = True
judgeenv.problem_globs = [os.path.join(ROOT_DIR, 'dmoj_problems/*')]


# Must come after judgeenv.env.update(config)
import dmoj
from dmoj.executors.CPP20 import Executor as CPP20Executor
from dmoj.executors.CPP17 import Executor as CPP17Executor
from dmoj.executors.CPP14 import Executor as CPP14Executor
from dmoj.executors.CPP11 import Executor as CPP11Executor
from dmoj.executors.CPP03 import Executor as CPP03Executor
from dmoj.executors.RUST import Executor as RustExecutor
from dmoj.cptbox.filesystem_policies import RecursiveDir
from dmoj.problem import Problem
from dmoj.utils.load import load_modules

load_modules(
    ['C', 'CLANG', 'CLANGX', 'CPP03', 'CPP11', 'CPP14', 'CPP17', 'CPP20', 'PY3', 'RUST'],
    dmoj.executors.load_executor,
    'Executor',
    dmoj.executors.executors,
    dmoj.executors._unsupported_executors,
    loading_message='Skipped self-tests' if judgeenv.skip_self_test else 'Self-testing executors',
)

# The checker uses standard C++ executors, so we must 
# whitelist OpenSSL for them too, or the checker crashes.
# Patch the standard executors in-place
permissions = [
    RecursiveDir(OPENSSL_INCLUDE_PATH),
    RecursiveDir(GCC_ROOT_PATH),  # Allow reading the compiler binary and its internal headers
    RecursiveDir(OPENSSL_LIB_PATH),
    RecursiveDir(ALL_SOFTWARE_PATH)
]


if ENABLE_FORCE_GCC:
    # Define a helper function to force the command path
    def _force_cmd(cmd_path: str):
        def _get_command(self):
            return cmd_path
        return _get_command

    for ExecutorClass in [CPP20Executor, CPP17Executor, CPP14Executor, CPP11Executor]:
        # 1. Add Permissions (solves Permission Denied)
        ExecutorClass.fs.extend(permissions)
        
        # 2. Patch get_command (solves AssertionError and -std=c++20 error)
        # This ensures that when the class is instantiated, self.command is set to GCC_BIN
        # 3. Explicitly set command on the class too (safety net)
        ExecutorClass.command = GCC_BIN
        ExecutorClass.get_command = _force_cmd(GCC_BIN)
        
    print(f"DEBUG: Patched C++ Executors to force compiler: {GCC_BIN}")

from dmoj import contrib
contrib.load_contrib_modules()

class ParlayLibCPPExecutor(CPP17Executor):
    name = 'CPP_PARLAY'
    fs = CPP17Executor.fs + [RecursiveDir(parlaylib_include_path), RecursiveDir(ALL_SOFTWARE_PATH)]
    nproc = -1 # unlimited processes to allow creating threads correctly
    
    def get_compile_args(self):
        parent_args = super().get_compile_args()
        custom_flags = [f"-I{parlaylib_include_path}", "-pthread", "-fno-diagnostics-color"]
        return parent_args[:-3] + custom_flags + parent_args[-3:]


if LANGUAGE == "rust":
    # We Define a clean Cargo.toml.
    # We REMOVE 'dmoj', 'libc', 'rand' so Cargo doesn't crash looking for them.
    MINIMAL_CARGO_TOML = b"""\
    [package]
    name = "user_submission"
    version = "1.0.0"
    edition = "2021"

    [dependencies]
    # No dependencies listed here.
    # Rayon is linked manually via flags below.
    """

    class RustRayonExecutor(RustExecutor):
        name = 'Rust_Rayon'
        fs = RustExecutor.fs + [
            RecursiveDir(host_deps_folder),  # Rayon deps
            RecursiveDir(OPENSSL_LIB_PATH),  # Needed by cargo/rustc
            RecursiveDir(OPENSSL_INCLUDE_PATH),
            RecursiveDir(RUST_LIB_PATH),     # Needed by cargo/rustc
            RecursiveDir(GCC_ROOT_PATH)       # Needed by cargo/rustc
        ]
        nproc = -1

        def create_files(self, problem_id, source_code, *args, **kwargs) -> None:
            super().create_files(problem_id, source_code, *args, **kwargs)

            with open(self._file('Cargo.toml'), 'wb') as f:
                f.write(MINIMAL_CARGO_TOML)


        def get_compile_args(self):
            args = super().get_compile_args()

            # 2. Switch 'build' to 'rustc' so we can pass custom flags
            if 'build' in args:
                args[args.index('build')] = 'rustc'

            # 3. Append our linker flags after the '--' separator
            # This tells Cargo: "Pass everything after this line directly to the compiler"
            args.extend([
                '--', 
                '-L', f"dependency={SANDBOX_FOLDER}",
                '--extern', f"rayon={SANDBOX_RAYON_FILE}"
            ])
            
            return args
    

class Judge:
    """Judge for evaluating C++ solutions with custom executors"""
    
    def __init__(self, problem_id: str, time_limit: int = 2, memory_limit: int = 262144, language: str = "cpp"):
        self.problem_id = problem_id
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.language = language  # "cpp" or "rust"
        self.problem = Problem(problem_id, time_limit, memory_limit, meta={})
        self.temp_dir = tempfile.mkdtemp(prefix='judge_')
        # os.environ['PARLAY_NUM_THREADS'] = f"{thread_number}"

    def get_all_test_cases(self):
        """Flatten batched and regular test cases"""
        for item in self.problem.cases():
            # Check if it's a BatchedTestCase by looking for 'batched_cases' attribute
            if hasattr(item, 'batched_cases'):
                # It's a batch, yield individual cases
                for case in item.batched_cases:
                    yield case
            else:
                # It's a regular TestCase
                yield item
        
    def evaluate(self, source_code: str, repeat: int = 1, display_result: bool = False) -> Dict[str, Any]:
        """
        Evaluate a C++ solution and return results.
        """
        # Compile the solution
        try:
            if self.language == "cpp":
                executor = ParlayLibCPPExecutor(self.problem_id, source_code.encode('utf-8'))
                binary_path = executor._file(executor.problem)
            elif self.language == "rust":
                executor = RustRayonExecutor(self.problem_id, source_code.encode("utf-8"))
                # Force compilation (some executors compile lazily)
                if hasattr(executor, "compile"):
                    executor.compile()
                binary_path = executor.get_compiled_file() if hasattr(executor, "get_compiled_file") else executor.get_cmdline()[0]
            else:
                raise ValueError(f"Unsupported language: {self.language}")
        except Exception as e:
            if display_result:
                print(e)
            return {
                'passed': False,
                'compile_error': str(e),
                'total_tests': 0,
                'passed_tests': 0,
                'results': [],
                'avg_time': 0.0,
                'max_time': 0.0,
                'total_time': 0.0,
                'variance_time': 0.0,
                'feedback': "Compilation Error."
            }
        
        # First run to check if all tests pass
        total_time = 0.0
        max_time = 0.0
        passed_count = 0
        passed = True
        total_tests = 0
        feedback = ''
        
        cases = list(self.get_all_test_cases())
        for i, case in enumerate(cases, 1):
            total_tests += 1
            result = self._run_test_case(binary_path, case, i)
            # print("result: ", result)
            if display_result:
                status = "✓" if result['passed'] else "✗"
                print(f"Test {result['test_num']}: {status} ({result['time']:.4f}s)")
                if not result['passed']:
                    print(f"  Expected length: {len(result['expected'])}")
                    print(f"  Got length:      {len(result['actual'])}")
                    print(f"  Expected: {result['expected'][:100]}")
                    print(f"  Got:      {result['actual'][:100]}")
                    print(f"  Feedback: {result['feedback']}")

            # if not result['passed']:
            #     # add to failed test info
            #     failed_test_info = f"Test {result['test_num']}: Expected length: {len(result['expected'])}\t"
            #     failed_test_info += f"Got length:      {len(result['actual'])}\t"
            #     failed_test_info += f"Expected: {result['expected'][:100]}\t"   
            #     failed_test_info += f"Got:      {result['actual'][:100]}"                        
                                                                
                                                                
                # failed_tests_info.append(failed_test_info)
                    
            
            if result['passed']:
                passed_count += 1
            else:
                feedback = result['feedback']
                passed = False
                break
            
            total_time += result['time']
            max_time = max(max_time, result['time'])
        
        # If all tests passed and repeat > 1, run additional times
        all_run_times = [total_time]
        
        if passed and repeat > 1:
            if display_result:
                print(f"\nAll tests passed! Running {repeat - 1} more time(s) for accurate timing...")
            
            for run_num in range(2, repeat + 1):
                run_total_time = 0.0
                run_max_time = 0.0
                
                for i, case in enumerate(self.get_all_test_cases(), 1):
                    result = self._run_test_case(binary_path, case, i)
                    run_total_time += result['time']
                    run_max_time = max(run_max_time, result['time'])
                
                all_run_times.append(run_total_time)
                max_time = max(max_time, run_max_time)
                
                if display_result:
                    print(f"Run {run_num}/{repeat}: {run_total_time:.4f}s total")
            
            total_time = sum(all_run_times)
            avg_total_time = total_time / repeat
            variance_time = sum((t - avg_total_time) ** 2 for t in all_run_times) / repeat
            
            if display_result:
                print(f"\nTiming statistics over {repeat} runs:")
                print(f"  Average: {avg_total_time:.4f}s")
                print(f"  Std Dev: {variance_time ** 0.5:.4f}s")
                print(f"  Min: {min(all_run_times):.4f}s")
                print(f"  Max: {max(all_run_times):.4f}s")
        else:
            variance_time = 0.0
        
        return {
            'passed': passed,
            'passed_tests': passed_count,
            'total_tests': total_tests,
            'avg_time': (total_time / repeat) / total_tests if total_tests > 0 else 0.0,
            'max_time': max_time,
            'total_time': total_time / repeat,
            'variance_time': variance_time,
            'feedback': feedback
        }
    
    def _run_test_case(self, binary_path: str, case, test_num: int) -> Dict[str, Any]:
        """Run a single test case using file-based I/O with precise CPU time measurement"""
        # Create temp files for input/output
        input_file = os.path.join(self.temp_dir, f'input_{test_num}.txt')
        output_file = os.path.join(self.temp_dir, f'output_{test_num}.txt')
        
        # Write input to file
        with open(input_file, 'wb') as f:
            f.write(case.input_data())
        
        start_time = time.perf_counter()
        stderr_output = ""
        feedback = ''
    
        try:
            env = os.environ.copy()
            with open(input_file, 'rb') as stdin_f, open(output_file, 'wb') as stdout_f:
                proc = subprocess.Popen(
                    [binary_path],
                    stdin=stdin_f,
                    stdout=stdout_f,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                
                try:
                    # proc.wait(timeout=self.time_limit * 3)
                    # communicate returns (stdout, stderr). Since stdout is a file, the first item is None.
                    _, stderr_bytes = proc.communicate(timeout=self.time_limit * 3)
                    exec_time = time.perf_counter() - start_time
                    returncode = proc.returncode
                    
                    if stderr_bytes:
                        stderr_output = stderr_bytes.decode("utf-8", errors="replace")

                    # exec_time = time.perf_counter() - start_time  # Wall-clock time
                    # returncode = proc.returncode
                    # stderr_bytes = out_err[1] if out_err and len(out_err) > 1 else b""

                    
                except subprocess.TimeoutExpired:
                    proc.kill()
                    _, stderr_bytes = proc.communicate()
                    # proc.wait()
                    exec_time = self.time_limit * 3
                    returncode = -1
                    if stderr_bytes:
                        stderr_output = stderr_bytes.decode("utf-8", errors="replace")
                    feedback = "Time Limit Exceeded."
            
            with open(output_file, 'rb') as f:
                output = f.read()
                
        except Exception as e:
            exec_time = 0.0
            output = b''
            returncode = -1
            stderr_output = str(e)
        
        # Clean up temp files
        try:
            os.remove(input_file)
            os.remove(output_file)
        except:
            pass
        
        # Use DMOJ's checker
        checker_fn = case.checker()
        
        check_result = checker_fn(
            problem_id=self.problem_id,
            case=case,
            process_output=output,
            judge_output=case.output_data(),
            judge_input=case.input_data(),
            point_value=case.points,
            execution_time=exec_time,
        )
        if len(feedback) == 0:
            feedback = "Wrong Answer."
        if isinstance(check_result, CheckerResult):
            passed = check_result.passed
            if hasattr(check_result, 'feedback') and check_result.feedback:
                feedback += check_result.feedback
        else:
            passed = bool(check_result)
        
        
        # print("passed: ")
        # print("passed: ", passed)
        # print("stderr_output: ", stderr_output)
        if not passed and stderr_output:
            print(f"Test {test_num} FAILED. Stderr captured:\n{stderr_output[:1000]}")
            # Optional: Append to feedback so it shows in the summary
            feedback += f"\nSTDERR: {stderr_output[:500]}"

        return {
            'test_num': test_num,
            'passed': passed,
            'time': exec_time,
            'expected': case.output_data().decode('utf-8', errors='replace').strip(),
            'actual': output.decode('utf-8', errors='replace').strip(),
            'memory': 0,
            'feedback': feedback
        }


if __name__ == "__main__":
    problem_id = "coci08c6p5"
    solution_path = './solution.cpp'
    # Read solution from file
    with open(os.path.join(ROOT_DIR, solution_path), 'r') as f:
        source_code = f.read()
    
    # Create judge and evaluate
    judge = Judge(problem_id, time_limit=2, memory_limit=262144, language=LANGUAGE)
    results = judge.evaluate(source_code, repeat = 5, display_result=True)
    
    # Print results
    print(f"Problem: {problem_id}")
    print(f"Status: {'PASSED' if results['passed'] else 'FAILED'}")
    print(f"Tests: {results['passed_tests']}/{results['total_tests']}")
    print(f"Avg Time: {results['avg_time'] * 1000:.2f}ms")
    print(f"Max Time: {results['max_time'] * 1000:.2f}ms")
    print(f"Variance Time: {results['variance_time'] * 1000:.2f}ms")
    print(f"Total Time: {results['total_time'] * 1000:.2f}ms")
    print()
