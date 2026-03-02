import os
import shutil
import subprocess
import logging
from os import PathLike
from drivers.driver_wrapper import DriverWrapper, BuildOutput, RunOutput, GeneratedTextResult
from util import run_command

class RustDriverWrapper(DriverWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Determine where the base Cargo project is located
        self.rust_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rust")
        self.built_timeout = 90
        self.run_timeout = 360

    def write_source(self, content: str, fpath: PathLike) -> bool:
        """ Write the given c++ source to the given file. """
        with open(fpath, "w") as fp:
            fp.write(content)
        return True

    def compile(self, *binaries: PathLike, output_path: PathLike = "release", **kwargs) -> BuildOutput:
        """ Runs cargo build """
        # binaries arg is ignored as Cargo handles sources
        cmd = f"cargo build --release --manifest-path {os.path.join(output_path, 'Cargo.toml')}"
        run_env = kwargs.get('env', os.environ.copy())
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.build_timeout, env=run_env)
            return BuildOutput(res.returncode, res.stdout, res.stderr)
        except subprocess.TimeoutExpired as e:
            return BuildOutput(-1, "", f"[Timeout] {str(e)}")

    def run(self, executable: PathLike, **run_config) -> RunOutput:
        """ executable here is the path to the cargo binary """
        launch_format = self.launch_configs["format"]
        launch_cmd = launch_format.format(exec_path=executable, args="", **run_config).strip()
        # print("launch_cmd: ", launch_cmd)
        # launch_cmd = f"{executable} {run_config.get('args', '')}"
        try:
            run_process = run_command(launch_cmd, timeout=self.run_timeout, dry=self.dry)
        except subprocess.TimeoutExpired as e:
            return RunOutput(-1, str(e.stdout), f"[Timeout] {str(e.stderr)}", config=run_config)
        except UnicodeDecodeError as e:
            logging.warning(f"UnicodeDecodeError: {str(e)}\nRunnning command: {launch_cmd}")
            return RunOutput(-1, "", f"UnicodeDecodeError: {str(e)}", config=run_config)
        return RunOutput(run_process.returncode, run_process.stdout, run_process.stderr, config=run_config)

        # try:
        #     res = subprocess.run(launch_cmd, shell=True, capture_output=True, text=True, timeout=self.run_timeout)
        #     return RunOutput(res.returncode, res.stdout, res.stderr, config=run_config)
        # except subprocess.TimeoutExpired as e:
        #     return RunOutput(-1, str(e.stdout), f"[Timeout] {str(e.stderr)}", config=run_config)


    def test_single_output(self, prompt: str, output: str, test_driver_file: PathLike, problem_size: str) -> GeneratedTextResult:
        """ 
        1. Copies the Rust harness to scratch dir.
        2. Reads the benchmark specific driver (test_driver_file).
        3. Injects the LLM output into it.
        4. Writes it to src/main.rs.
        5. Compiles and Runs.
        """
        import tempfile
        
        # We need a temp dir that acts as the Cargo Project
        logging.debug(f"Testing output:\n{output}")
        with tempfile.TemporaryDirectory(dir=self.scratch_dir) as tmpdir:
            # 1. Setup Cargo Project
            shutil.copy(os.path.join(self.rust_root, "Cargo.toml"), tmpdir)
            os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
            shutil.copy(os.path.join(self.rust_root, "src", "lib.rs"), os.path.join(tmpdir, "src", "lib.rs"))

            # 2. Prepare Benchmark Source
            # We assume test_driver_file (e.g. convex_hull.rs) has a placeholder // LLM_OUTPUT_HERE
            with open(test_driver_file, 'r') as f:
                driver_code = f.read()
            
            # 3. Inject Code
            # Simple injection: append code or replace a specific token. 
            # Ideally your benchmark files have `mod generated { // LLM_OUTPUT_HERE }`
            final_source = driver_code.replace("// LLM_OUTPUT_HERE", output)

            # 4. Write to main.rs
            main_rs_path = os.path.join(tmpdir, "src", "main.rs")
            write_success = self.write_source(final_source, main_rs_path)
            # with open(main_rs_path, "w") as f:
            #     f.write(final_source)
            logging.debug(f"Wrote source to {main_rs_path}.")
            # print("Written Rust source to:", main_rs_path)
            # print("final_source: ", final_source)

            # 5. Compile
            build_env = os.environ.copy()
            # Set the flag equivalent to -DDRIVER_PROBLEM_SIZE
            build_env["DRIVER_PROBLEM_SIZE"] = str(eval(str(problem_size)))

            # Pass this 'build_env' to your compile method
            # Note: You may need to update self.compile definition to accept 'env'
            build_result = self.compile(output_path=tmpdir, env=build_env)
            # print("build_result: ", build_result)
            logging.debug(f"Build result: {build_result}")
            if self.display_build_errors and build_result.stderr and not build_result.did_build:
                logging.debug(build_result.stderr)

            
            # 6. Run
            run_results = []
            if build_result.did_build:
                exe_path = os.path.join(tmpdir, "target", "release", "pareval_runner")
                configs = self.launch_configs["params"]
                # print("configs: ", configs)
                for c in configs:
                    run_result = self.run(exe_path, **c)
                    run_results.append(run_result)
                    if self.display_runs:
                        logging.debug(run_result.stderr)
                        logging.debug(run_result.stdout)
                    if self.early_exit_runs and (run_result.exit_code != 0 or not run_result.is_valid):
                        break
            else:
                run_results = None
            logging.debug(f"Run result: {run_results}")
            if run_results:
                for run_result in run_results:
                    if run_result.exit_code != 0:
                        logging.debug(f"Ouputs:\n\tstdout: {run_result.stdout}\n\tstderr: {run_result.stderr}")

            return GeneratedTextResult(write_success, build_result, run_results)