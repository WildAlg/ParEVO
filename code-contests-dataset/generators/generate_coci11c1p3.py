import random
import sys

def solve_fast(n, numbers):
    """
    Solves the Planet X3 problem efficiently using the bitwise contribution logic.
    Time Complexity: O(N * log(max_val)) - effectively O(N) given the constraints.
    """
    total_friendship_value = 0
    # Max value is < 1,000,000, which is less than 2^20.
    # We check bits 0 to 19.
    max_bits = 20 
    
    # Iterate through each bit position
    for i in range(max_bits):
        count_ones = 0
        
        # Count how many numbers have the i-th bit set
        for num in numbers:
            if (num >> i) & 1:
                count_ones += 1
        
        count_zeros = n - count_ones
        
        # The contribution of this bit position to the total sum is:
        # (number of pairs with different bits) * (value of this bit)
        # Pairs with different bits = count_ones * count_zeros
        contribution = count_ones * count_zeros * (1 << i)
        total_friendship_value += contribution
        
    return total_friendship_value

def generate_test_case(n_size, input_filename, output_filename):
    """
    Generates a test case with n_size residents.
    Writes the input to input_filename.
    Calculates the solution and writes it to output_filename.
    """
    print(f"Generating test case with N = {n_size}...")
    
    # Generate N random integers between 1 and 999,999
    # The problem states names are positive integers smaller than 1,000,000
    residents = [random.randint(1, 999999) for _ in range(n_size)]
    
    # Write Input File
    print(f"Writing inputs to {input_filename}...")
    with open(input_filename, 'w') as f_in:
        f_in.write(f"{n_size}\n")
        # Joining list of strings is faster than writing line by line for large N
        f_in.write('\n'.join(map(str, residents)) + '\n')
    
    # Calculate Expected Output
    print("Calculating solution (this might take a moment)...")
    result = solve_fast(n_size, residents)
    
    # Write Output File
    print(f"Writing expected output to {output_filename}...")
    with open(output_filename, 'w') as f_out:
        f_out.write(f"{result}\n")
        
    print("Done.")

if __name__ == "__main__":
    # === PARAMETER SETTINGS ===
    # The problem has linear time complexity O(N).
    # To make the test case take ~10x more time than a 'typical' smaller case (e.g., N=100,000),
    # we set N to the maximum constraint allowed by the problem statement.
    #
    # Parameter: N (Number of residents)
    # Value: 1,000,000 (The maximum constraint)
    #
    # Note: If you specifically wanted to force the Standard Solution (which is very fast) 
    # to run 10x slower than the time limit typically allotted for N=10^6, you would 
    # theoretically need N=10^7. However, that violates the problem constraints 
    # (N <= 1,000,000). We use the maximum valid N here to generate the stress test.
    TEST_CASE_SIZE = 1_000_000
    
    for i in range(1, 11):
        INPUT_FILE = f"gen.{i}.in"
        OUTPUT_FILE = f"gen.{i}.out"
        with open("init.yml", "a") as f:
            f.write(f"""- {{in: gen.{i}.in, out: gen.{i}.out, points: 1}}\n""")
        generate_test_case(TEST_CASE_SIZE, INPUT_FILE, OUTPUT_FILE)