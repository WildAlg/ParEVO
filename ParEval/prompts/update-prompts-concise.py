import os
import argparse

RAW_DIR = "raw"
TARGET = "compute in parallel."
INSERT = " Be concise and do not include extensive comments."

def add_insert():
    for root, _, files in os.walk(RAW_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            with open(fpath, "r") as f:
                content = f.read()
            if TARGET in content and INSERT.strip() not in content:
                new_content = content.replace(TARGET, TARGET + INSERT)
                with open(fpath, "w") as f:
                    f.write(new_content)
                print(f"Added INSERT to: {fpath}")

def remove_insert():
    for root, _, files in os.walk(RAW_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            with open(fpath, "r") as f:
                content = f.read()
            if INSERT in content:
                new_content = content.replace(INSERT, "")
                with open(fpath, "w") as f:
                    f.write(new_content)
                print(f"Removed INSERT from: {fpath}")

def convert_omp_to_cilk():
    for root, _, files in os.walk(RAW_DIR):
        for fname in files:
            if fname == "omp":
                fpath = os.path.join(root, fname)
                with open(fpath, "r") as f:
                    content = f.read()
                new_content = content.replace("OpenMP", "OpenCilk")
                cilk_path = os.path.join(root, "cilk")
                with open(cilk_path, "w") as f:
                    f.write(new_content)
                print(f"Created: {cilk_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["remove_insert", "insert", "convert_cilk"], help="Remove the INSERT string from files, insert it, or convert omp to cilk.")
    args = parser.parse_args()

    if args.mode == "remove_insert":
        remove_insert()
    elif args.mode == "insert":
        add_insert()
    elif args.mode == "convert_cilk":
        convert_omp_to_cilk()