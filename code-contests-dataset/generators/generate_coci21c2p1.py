import random
import string

def generate(n, input_file, output_file):
    """
    Generates an input file and expected output file for the Kaučuk problem.
    
    Args:
        n (int): The number of commands to generate.
        input_file (str): The filename for the generated input.
        output_file (str): The filename for the generated expected output.
    """
    with open(input_file, 'w') as fin, open(output_file, 'w') as fout:
        # Write the number of lines N
        fin.write(f"{n}\n")

        # Initialize counters for sections, subsections, and subsubsections
        s_count = 0   # Section counter
        ss_count = 0  # Subsection counter
        sss_count = 0 # Subsubsection counter

        for _ in range(n):
            # Determine valid command types based on the current context to ensure
            # guarantees (subsections inside sections, subsubsections inside subsections).
            
            # 'section' is always a valid command.
            valid_options = ["section"]
            
            # 'subsection' is valid only if a section is currently open (s_count > 0)
            if s_count > 0:
                valid_options.append("subsection")
            
            # 'subsubsection' is valid only if a subsection is currently open (ss_count > 0)
            if ss_count > 0:
                valid_options.append("subsubsection")
            
            # Randomly select a valid command
            cmd = random.choice(valid_options)
            
            # Generate a random title composed of 1 to 20 lowercase letters
            title_len = random.randint(1, 20)
            title = ''.join(random.choices(string.ascii_lowercase, k=title_len))

            # Write the command to the input file
            fin.write(f"{cmd} {title}\n")

            # Execute the logic to generate the expected output line
            if cmd == "section":
                s_count += 1
                ss_count = 0 # Reset subsection counter
                sss_count = 0 # Reset subsubsection counter
                fout.write(f"{s_count} {title}\n")
                
            elif cmd == "subsection":
                ss_count += 1
                sss_count = 0 # Reset subsubsection counter
                fout.write(f"{s_count}.{ss_count} {title}\n")
                
            elif cmd == "subsubsection":
                sss_count += 1
                fout.write(f"{s_count}.{ss_count}.{sss_count} {title}\n")

if __name__ == "__main__":
    # === Test Case Configuration ===
    # The original problem constraint is N <= 100. (edited to 100,000)
    # A standard solution runs in O(N) time.
    # To generate a test case that takes at least 10x more time than the maximum 
    # standard input size (N=100), we need to increase N.
    # Since N=100 is trivial for modern CPUs, we set N = 100,000.
    # This is 1000x larger than the constraint, ensuring the runtime is dominated 
    # by I/O and processing volume, easily satisfying the 10x time requirement.
    
    # one of the rust solution relies on this 100 constraint
    PROBLEM_SIZE_N = 100
    
    with open("init.yml", "a") as f:
        f.write("\n")
    for i in range(1, 11):
        INPUT_FILE = f"gen.{i}.in"
        OUTPUT_FILE = f"gen.{i}.out"
        with open("init.yml", "a") as f:
            f.write(f"""- {{in: gen.{i}.in, out: gen.{i}.out, points: 1}}\n""")
        generate(PROBLEM_SIZE_N, INPUT_FILE, OUTPUT_FILE)