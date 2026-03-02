import os
import re
import argparse

def process_cpu_file(cpu_path):
    parlay_path = os.path.join(os.path.dirname(cpu_path), "parlay.cc")

    with open(cpu_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    inserted = False
    i = 0

    # Copy initial comment block at top (// or /* ... */)
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped == "":
            new_lines.append(line)
            i += 1
        else:
            break

    # Insert parlay include before first #include
    new_lines.append('#include <parlay/primitives.h>\n')
    inserted = True

    # Copy rest of the file
    new_lines.extend(lines[i:])

    # Replace std::vector -> parlay::sequence
    content = "".join(new_lines)
    content = re.sub(r"\bstd::vector\b", "parlay::sequence", content)

    with open(parlay_path, "w") as f:
        f.write(content)

    print(f"Processed: {cpu_path} -> {parlay_path}")


def traverse_benchmarks(root="cpp/benchmarks"):
    for dirpath, _, filenames in os.walk(root):
        if "cpu.cc" in filenames:
            cpu_path = os.path.join(dirpath, "cpu.cc")
            process_cpu_file(cpu_path)

# ========== Create Parlay Correct Baseline ===========
def process_baseline_file(baseline_path):
    with open(baseline_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    functions_for_parlay = []
    inside_function = False
    brace_count = 0
    function_buffer = []

    # Step 1: Copy comments at top
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped == "":
            new_lines.append(line)
            i += 1
        else:
            break

    # Step 2: Insert conditional include
    new_lines.append("#ifdef USE_PARLAY\n")
    new_lines.append("#include <parlay/primitives.h>\n")
    new_lines.append("#endif\n\n")

    # Step 3: Scan for functions and collect duplicates
    for line in lines[i:]:
        new_lines.append(line)

        # Match function signatures (simplified: look for lines ending with '{' and containing '(' and ')')
        if not inside_function and re.search(r"\)\s*\{", line):
            inside_function = True
            brace_count = 0
            function_buffer = []

        if inside_function:
            function_buffer.append(line)
            brace_count += line.count("{") - line.count("}")
            if brace_count == 0:
                # End of function
                inside_function = False
                func_text = "".join(function_buffer)
                parlay_version = re.sub(r"\bstd::vector\b", "parlay::sequence", func_text)
                functions_for_parlay.append(parlay_version)
                function_buffer = []

    # Step 4: Append parlay-wrapped versions
    if functions_for_parlay:
        new_lines.append("\n#ifdef USE_PARLAY\n")
        new_lines.extend(functions_for_parlay)
        new_lines.append("#endif\n")

    # Step 5: Save back
    with open(baseline_path, "w") as f:
        f.writelines(new_lines)

    print(f"Processed: {baseline_path}")


def create_correct_parlay_templates(root="cpp/benchmarks"):
    for dirpath, _, filenames in os.walk(root):
        if "baseline.hpp" in filenames:
            baseline_path = os.path.join(dirpath, "baseline.hpp")
            process_baseline_file(baseline_path)

# ======== Undo ========
def apply_parlay_modifications(baseline_path):
    """Apply Parlay modifications to baseline.hpp."""
    with open(baseline_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    inside_cuda_block = False

    for line in lines:
        # Skip CUDA-specific blocks
        if line.strip().startswith("#if defined(USE_CUDA)"):
            inside_cuda_block = True
            continue
        if inside_cuda_block and line.strip().startswith("#endif"):
            inside_cuda_block = False
            continue
        if inside_cuda_block:
            continue

        new_lines.append(line)

    # Add Parlay include at the top (after other includes)
    insert_index = 0
    for i, line in enumerate(new_lines):
        if line.strip().startswith("#include"):
            insert_index = i + 1
    new_lines.insert(insert_index, "#ifdef USE_PARLAY\n#include <parlay/primitives.h>\n#endif\n")

    # Add duplicate block wrapped in #ifdef USE_PARLAY … #endif
    functions_block = []
    for line in new_lines:
        if "std::vector" in line:
            functions_block.append(line.replace("std::vector", "parlay::sequence"))
        else:
            functions_block.append(line)

    new_lines.append("\n#ifdef USE_PARLAY\n")
    new_lines.extend(functions_block)
    new_lines.append("#endif\n")

    with open(baseline_path, "w") as f:
        f.writelines(new_lines)

    print(f"Applied Parlay modifications: {baseline_path}")


def undo_parlay_modifications(baseline_path):
    """Undo Parlay modifications in a baseline.hpp file."""
    with open(baseline_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    inside_parlay_block = False
    i = 0
    while i < len(lines):
        line = lines[i]

        # Remove the inserted #ifdef USE_PARLAY include
        if line.strip() == "#ifdef USE_PARLAY":
            if (
                i + 2 < len(lines)
                and lines[i + 1].strip() == "#include <parlay/primitives.h>"
                and lines[i + 2].strip() == "#endif"
            ):
                i += 3
                continue

        # Remove the duplicated function block
        if line.strip() == "#ifdef USE_PARLAY":
            inside_parlay_block = True
            i += 1
            continue
        if inside_parlay_block:
            if line.strip() == "#endif":
                inside_parlay_block = False
            i += 1
            continue

        # Otherwise, keep line
        new_lines.append(line)
        i += 1

    with open(baseline_path, "w") as f:
        f.writelines(new_lines)

    print(f"Undo complete: {baseline_path}")


def main():
    parser = argparse.ArgumentParser(description="Modify or undo Parlay changes in baseline.hpp files.")
    parser.add_argument("directory", default="cpp/benchmarks", help="Root directory to search (e.g., cpp/benchmarks)")
    parser.add_argument("--mode", choices=["apply", "undo"], required=True, help="Apply or undo modifications")

    args = parser.parse_args()

    for root, _, files in os.walk(args.directory):
        if "baseline.hpp" in files:
            baseline_path = os.path.join(root, "baseline.hpp")
            if args.mode == "apply":
                apply_parlay_modifications(baseline_path)
            elif args.mode == "undo":
                undo_parlay_modifications(baseline_path)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Transform baseline.hpp files to add optional Parlay versions of functions.")
#     parser.add_argument("root", type=str, nargs="?", default="cpp/benchmarks", help="Root directory to traverse (default: cpp/benchmarks)")
#     args = parser.parse_args()

#     create_correct_parlay_templates(args.root)

