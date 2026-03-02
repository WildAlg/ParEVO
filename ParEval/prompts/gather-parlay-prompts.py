import json
import argparse

def main(args):

    # Load the input JSON file
    with open(args.input_file, "r") as infile:
        data = json.load(infile)

    # Filter entries where parallelism_model is "parlay" or "cilk"
    filtered = [
        entry for entry in data
        if entry.get("parallelism_model") in {"parlay", "cilk", "omp"}
    ]

    # Save to the output file
    with open(args.output_file, "w") as outfile:
        json.dump(filtered, outfile, indent=2)

    print(f"Extracted {len(filtered)} entries to '{args.output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSON entries by parallelism_model.")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to save the filtered JSON output")
    args = parser.parse_args()
    main(args)
