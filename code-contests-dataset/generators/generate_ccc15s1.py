import random

INPUT_FILE = ""
OUTPUT_FILE = ""

def generate(k):
    """
    Generates a test case for CCC 2015 Senior #1.
    
    Args:
        k (int): The number of integers the boss will say.
    """
    
    commands = []
    
    # We need to track the 'virtual' stack size during generation 
    # to ensure we don't produce a "0" command when the stack is empty.
    # The problem guarantees: "At any point, your boss will have said at least 
    # as many positive numbers as 'zero' statements."
    current_stack_size = 0
    
    for _ in range(k):
        # If stack is empty, we MUST push a positive number
        if current_stack_size == 0:
            val = random.randint(1, 100)
            commands.append(val)
            current_stack_size += 1
        else:
            # If stack is not empty, we can randomly choose to push or pop.
            # Let's verify we don't just pop everything immediately to keep the case interesting.
            # 40% chance to say "zero" (pop), 60% chance to say a number (push).
            if random.random() < 0.4:
                commands.append(0)
                current_stack_size -= 1
            else:
                val = random.randint(1, 100)
                commands.append(val)
                current_stack_size += 1
                
    # --- Write Input File ---
    with open(INPUT_FILE, 'w') as f:
        f.write(f"{k}\n")
        for num in commands:
            f.write(f"{num}\n")
            
    # --- Generate Expected Output ---
    # Solution Logic: Stack implementation as per editorial
    stack = []
    for x in commands:
        if x == 0:
            stack.pop()
        else:
            stack.append(x)
    
    result = sum(stack)
    
    # --- Write Output File ---
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"{result}\n")

if __name__ == "__main__":
    # The original problem constraint is K <= 100,000.(edited to 1,000,000)
    # The time complexity of the standard solution is O(K).
    # To ensure the test case takes ~10x more time, we multiply K by 10.
    # Parameter: K = 1,000,000

    for i in range(1, 11):
        INPUT_FILE = f"gen.{i}.in"
        OUTPUT_FILE = f"gen.{i}.out"
        with open("init.yml", "a") as f:
            f.write(f"""- {{in: gen.{i}.in, out: gen.{i}.out, points: 1}}\n""")
        
        generate(1_000_000)