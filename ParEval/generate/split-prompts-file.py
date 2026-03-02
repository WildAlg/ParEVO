import json
import glob
import argparse

def split_prompts_file(args):
    with open(args.file_name) as f:
        prompts = json.load(f)

    N = args.num_splits  # Number of splits
    chunk_size = (len(prompts) + N - 1) // N

    for i in range(N):
        with open(f"{args.file_name}.part{i}.json", "w") as f:
            json.dump(prompts[i*chunk_size:(i+1)*chunk_size], f, indent=2)

def merge_outputs(args):
    outputs = []
    for fname in sorted(glob.glob(f"{args.file_name}.part*.json")):
        with open(fname) as f:
            outputs.extend(json.load(f))

    with open(f"{args.file_name}.merged.json", "w") as f:
        json.dump(outputs, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["split", "merge"], required=True, help="Operation mode: split or merge")
    parser.add_argument("--file-name", type=str, required=True, help="Input prompts file")
    parser.add_argument("--num-splits", type=int, default=4, help="Number of splits to create")
    args = parser.parse_args()

    if args.mode == "split":
        split_prompts_file(args)
    elif args.mode == "merge":
        merge_outputs(args)
