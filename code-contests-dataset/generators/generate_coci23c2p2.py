import random
from collections import deque
import os, zipfile

INPUT_FILE = ""
OUTPUT_FILE = ""

def solve(n, start, end, grid):
    """
    Solves the problem using BFS.
    
    Parameters:
    n (int): Dimensions of the cube.
    start (tuple): (x, y, z) starting coordinates (1-indexed).
    end (tuple): (x, y, z) ending coordinates (1-indexed).
    grid (list): 3D list grid[z][x][y] (1-indexed) where 0 is empty and 1 is cloud.
    
    Returns:
    int: The minimum wing flaps or -1 if unreachable.
    """
    sx, sy, sz = start
    ex, ey, ez = end

    if grid[sz][sx][sy] == 1 or grid[ez][ex][ey] == 1:
        return -1

    queue = deque([(sx, sy, sz, 0)])
    visited = set()
    visited.add((sx, sy, sz))
    
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]

    while queue:
        cx, cy, cz, dist = queue.popleft()

        if (cx, cy, cz) == (ex, ey, ez):
            return dist

        for dx, dy, dz in directions:
            nx, ny, nz = cx + dx, cy + dy, cz + dz

            # Boundary checks (1-indexed)
            if 1 <= nx <= n and 1 <= ny <= n and 1 <= nz <= n:
                if grid[nz][nx][ny] == 0 and (nx, ny, nz) not in visited:
                    visited.add((nx, ny, nz))
                    queue.append((nx, ny, nz, dist + 1))
    
    return -1

def generate(n):
    """
    Generates a random test case of size n x n x n.
    Writes INPUT_FILE and OUTPUT_FILE.
    """
    # Generate Start and End positions (1-indexed)
    sx = random.randint(1, n)
    sy = random.randint(1, n)
    sz = 1  # Constraint: z_s = 1
    
    ex = random.randint(1, n)
    ey = random.randint(1, n)
    ez = random.randint(1, n)

    # Generate Grid (1-indexed): grid[z][x][y]
    # Index 0 is unused padding for cleaner 1-indexed access
    grid = [[[0 for _ in range(n + 1)] for _ in range(n + 1)] for _ in range(n + 1)]
    
    for z in range(1, n + 1):
        for x in range(1, n + 1):
            for y in range(1, n + 1):
                # Start and End positions cannot be clouds
                if (x, y, z) == (sx, sy, sz) or (x, y, z) == (ex, ey, ez):
                    grid[z][x][y] = 0
                else:
                    # ~20% chance of cloud
                    grid[z][x][y] = 1 if random.random() < 0.2 else 0

    # Write Input File
    with open(INPUT_FILE, "w") as f:
        f.write(f"{n}\n")
        f.write(f"{sx} {sy} {sz}\n")
        f.write(f"{ex} {ey} {ez}\n")
        
        # Output n matrices (one per height z from 1 to n)
        # Each matrix: rows correspond to x (1 to n), columns to y (1 to n)
        for z in range(1, n + 1):
            for x in range(1, n + 1):
                row_str = "".join(str(grid[z][x][y]) for y in range(1, n + 1))
                f.write(row_str + "\n")

    # Solve and Write Output
    result = solve(n, (sx, sy, sz), (ex, ey, ez), grid)
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"{result}\n")

if __name__ == "__main__":
    # The problem allows N up to 100. The complexity of the standard solution (BFS) is O(N^3).
    # By setting N = 100, we generate a grid with 1,000,000 nodes.
    # Compared to a small case (e.g., N=10, 1000 nodes), this maximizes the runtime 
    # (roughly 1000x more work) and queue memory usage, satisfying the requirement 
    # to stress the solution significantly more than trivial cases.
    for i in range(1, 11):
        # Map i to letters: 1->a, 2->b, ..., 10->j
        case_letter = chr(ord('a') + i - 1)
        INPUT_FILE = f"pingvin.in.5{case_letter}"
        OUTPUT_FILE = f"pingvin.out.5{case_letter}"
        # add another entry in case_points in init.yml
        generate(200)

    files = [f for f in os.listdir('.') if os.path.isfile(f) and not f.endswith(('.py', '.zip'))]

    with zipfile.ZipFile('pingvin.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f)

    print(f'Zipped {len(files)} files to pingvin.zip')

