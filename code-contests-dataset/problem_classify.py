import csv
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('GEMINI3_API_KEY')

# ------------------------ MODEL SETUP ------------------------



client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL_NAME = "gemini-3-pro-preview"


# ------------------------ CLASSIFIER FUNCTION ------------------------

def is_parallelizable_graph_problem(description: str) -> str:
    """
    Returns 'YES' or 'NO'
    """
    prompt = (
        "You are a classifier for computational graph problems. "
        "Given a competitive programming problem description, respond with "
        "ONLY 'YES' or 'NO'.\n\n"
        "Answer YES only if ALL of the following criteria are met:\n"
        "1. It is explicitly a graph problem (nodes/vertices and edges/connections)\n"
        "2. It involves computationally intensive graph operations (NOT simple adjacency checks or basic traversals)\n"
        "3. The problem has large-scale inputs (graphs with thousands to millions of nodes/edges)\n"
        "4. The algorithm is parallelizable - operations can be performed independently on different parts of the graph "
        "(e.g., finding connected components, parallel BFS/DFS on disconnected components, "
        "independent shortest path computations, parallel graph coloring)\n\n"
        "Answer NO if:\n"
        "- The problem has small test cases (n < 1000)\n"
        "- It's not fundamentally a graph problem (even if it mentions relationships)\n"
        "- The computation is trivial or can be solved with simple loops/lookups\n"
        "- The algorithm requires inherently sequential processing\n\n"
        f"Problem Description:\n{description}\n\n"
        "Response (YES or NO only):"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    answer = response.choices[0].message.content
    if answer is None:
        return "NO"

    answer = answer.strip().upper()
    if answer.startswith("YES"):
        return "YES"
    return "NO"


# ------------------------ LOAD INPUT CSV ------------------------

def load_problems(path="problems.csv"):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if len(row) >= 1:
                problem_id = row[0]
                description = row[1] if len(row) > 1 else ""
                solution = row[2] if len(row) > 2 else ""
                data.append({
                    "problem_id": problem_id,
                    "description": description,
                    "solution": solution
                })
    return pd.DataFrame(data)


# ------------------------ LOAD PREVIOUS RESULTS ------------------------

RESULTS_FILE = "results.csv"

def load_existing_results():
    if not os.path.exists(RESULTS_FILE):
        return {}
    df = pd.read_csv(RESULTS_FILE)
    return dict(zip(df.problem_id, df.result))


def append_result(problem_id, result):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["problem_id", "result"])
        writer.writerow([problem_id, result])


def write_positive_results(existing, out_file="results_YES.csv"):
    """
    Writes only the problem_ids with result == 'YES' into a new CSV file.
    """
    positive = [(pid, res) for pid, res in existing.items() if res.strip().upper() == "YES"]

    with open(out_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["problem_id", "result"])
        writer.writerows(positive)

    print(f"Wrote {len(positive)} positive results to {out_file}")

# ------------------------ MAIN PROCESS ------------------------

def main():
    problems_with_testcases = os.listdir('dmoj_problems')
    problems_with_testcases = {name.strip() for name in problems_with_testcases}

    df = load_problems("problems.csv")
    existing = load_existing_results()

    # Compute intersection
    csv_ids = set(df.problem_id)
    valid_ids = csv_ids.intersection(problems_with_testcases)

    print(f"Total problems in CSV: {len(df)}")
    print(f"Problems with testcases: {len(problems_with_testcases)}")
    print(f"Valid intersection: {len(valid_ids)}")
    print(f"Already classified: {len(existing)}")
    print("Starting classification...\n")

    for _, row in df.iterrows():
        pid = row.problem_id

        # Skip if not in intersection
        if pid not in valid_ids:
            continue

        # Skip if already classified
        if pid in existing:
            continue

        description = row.description

        # Query the model
        result = is_parallelizable_graph_problem(description)
        print(f"{pid}: {result}")

        # Save immediately
        append_result(pid, result)
    
    # After completed, do this to write down the positive results
    existing = load_existing_results()
    write_positive_results(existing=existing)



if __name__ == "__main__":
    main()
