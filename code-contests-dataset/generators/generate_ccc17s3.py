import random
import sys

# The problem has a unique deterministic output for any given input.
# The standard solution has a Time Complexity of O(N + L^2), where L is the maximum wood length (2000).
# Since L is small and fixed, the complexity is linear with respect to N for large N.

def solve(n, boards):
    """
    Solves the problem to generate the expected output using the standard frequency array approach.
    Complexity: O(N + L^2) where L=2000.
    """
    MAX_L = 2000
    # Frequency array to count occurrences of each wood length
    freq = [0] * (MAX_L + 1)
    for length in boards:
        freq[length] += 1
        
    max_fence_len = 0
    num_heights = 0
    
    # Iterate through all possible fence heights (sums of two wood lengths)
    # The smallest sum is 1+1=2, the largest is 2000+2000=4000
    for h in range(2, 2 * MAX_L + 1):
        current_len = 0
        
        # Calculate max boards for height h
        # We pair wood of length i with wood of length h-i
        # Iterate i from the smallest valid length up to h//2
        start_i = max(1, h - MAX_L)
        end_i = h // 2
        
        # Optimization: if no valid i exists, skip
        if start_i > end_i:
            continue
            
        for i in range(start_i, end_i + 1):
            j = h - i
            if i == j:
                # Pair pieces of same length
                current_len += freq[i] // 2
            else:
                # Pair pieces of different lengths
                current_len += min(freq[i], freq[j])
        
        # Update global maximums
        if current_len > max_fence_len:
            max_fence_len = current_len
            num_heights = 1
        elif current_len == max_fence_len:
            num_heights += 1
            
    return max_fence_len, num_heights

def generate(filename_in, filename_out, n, max_l):
    """
    Generates a test case with n pieces of wood with lengths up to max_l.
    Writes the input to filename_in and the solution to filename_out.
    """
    # Generate random wood lengths
    # Using random.choices is efficient for large datasets
    boards = random.choices(range(1, max_l + 1), k=n)
    
    # Write Input File
    with open(filename_in, 'w') as f:
        f.write(f"{n}\n")
        # Write space-separated integers
        f.write(" ".join(map(str, boards)))
        f.write("\n")
    
    # Generate Expected Output
    ans_len, ans_count = solve(n, boards)
    
    # Write Output File
    with open(filename_out, 'w') as f:
        f.write(f"{ans_len} {ans_count}\n")

if __name__ == "__main__":
    # === PARAMETERS ===
    # The problem specifies N <= 1,000,000.
    # The standard solution is O(N + L^2). Since L (2000) is a small constant, 
    # the runtime is dominated by reading input and processing N items.
    # To generate a test case that takes ~10x more time than the standard maximum case,
    # we set N to 10,000,000 (10 million).
    # While this is outside the problem's stated bounds (N=10^6), it fulfills the
    # requirement of stressing the standard solution by a factor of 10.
    
    TEST_CASE_SIZE = 1_000_000  # the max standard limit
    MAX_WOOD_LENGTH = 2000      # Fixed by problem statement
    
    
    for i in range(1, 11):
        INPUT_FILE = f"gen.{i}.in"
        OUTPUT_FILE = f"gen.{i}.out"
        # different format
        # with open("init.yml", "a") as f:
        #     f.write(f"""- {{in: gen.{i}.in, out: gen.{i}.out, points: 1}}\n""")
    
        print(f"Generating test case with N={TEST_CASE_SIZE}...")
        generate(INPUT_FILE, OUTPUT_FILE, TEST_CASE_SIZE, MAX_WOOD_LENGTH)
        print(f"Successfully generated '{INPUT_FILE}' and '{OUTPUT_FILE}'.")