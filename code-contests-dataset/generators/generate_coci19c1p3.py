"""
Test case generator for the Džumbus problem.

Problem: Given N friends with relaxation thresholds D_i, M pairs forming a forest,
and Q queries with total džumbus S_i, maximize people who exchange solutions.

Constraints:
- 0 <= M < N <= 1000
- 1 <= Q <= 2 * 10^5
- 1 <= D_i <= 10^9
- 1 <= S_i <= 10^9

The solution has O(N^2) preprocessing and O(Q * log(N)) per query (binary search).
To stress the solution, we want:
- N close to 1000 (maximizes DP computation)
- Q close to 200000 (maximizes query answering)
- Tree structure that creates many subtree combinations (e.g., balanced tree or chain)

Parameter to increase test case size: N (number of nodes) and Q (number of queries).
For a 10x slowdown from a small test, use N=1000, Q=200000 vs N~100, Q~20000.
"""

import random
import sys
from collections import defaultdict

INF = float('inf')


def generate_tree(n, tree_type="random"):
    """
    Generate a tree with n nodes (0-indexed internally, 1-indexed for output).
    Returns list of edges as (u, v) pairs (1-indexed).
    """
    if n <= 1:
        return []
    
    edges = []
    
    if tree_type == "random":
        # Random tree: each node connects to a random earlier node
        for i in range(1, n):
            parent = random.randint(0, i - 1)
            edges.append((parent + 1, i + 1))
    
    elif tree_type == "chain":
        # Chain: worst case for some DP approaches
        for i in range(1, n):
            edges.append((i, i + 1))
    
    elif tree_type == "star":
        # Star: all nodes connect to node 1
        for i in range(2, n + 1):
            edges.append((1, i))
    
    elif tree_type == "balanced":
        # Balanced binary tree: good for testing subtree merging
        for i in range(1, n):
            parent = (i - 1) // 2
            edges.append((parent + 1, i + 1))
    
    elif tree_type == "caterpillar":
        # Caterpillar: a chain with leaves hanging off
        # First half forms the spine
        spine_len = max(2, n // 2)
        for i in range(1, spine_len):
            edges.append((i, i + 1))
        # Remaining nodes attach to spine nodes
        for i in range(spine_len, n):
            spine_node = random.randint(1, spine_len)
            edges.append((spine_node, i + 1))
    
    return edges


def generate_forest(n, m, tree_type="random"):
    """
    Generate a forest with n nodes and m edges (m < n).
    """
    if m >= n:
        raise ValueError("m must be less than n for a forest")
    
    if m == 0:
        return []
    
    # Number of trees = n - m
    num_trees = n - m
    
    # Distribute nodes among trees
    # Each tree needs at least 1 node
    tree_sizes = [1] * num_trees
    remaining = n - num_trees
    
    # Randomly distribute remaining nodes
    for _ in range(remaining):
        tree_sizes[random.randint(0, num_trees - 1)] += 1
    
    edges = []
    node_offset = 0
    
    for size in tree_sizes:
        if size >= 2:
            tree_edges = generate_tree(size, tree_type)
            # Adjust node indices
            for u, v in tree_edges:
                edges.append((u + node_offset, v + node_offset))
        node_offset += size
    
    return edges


def solve(n, d, edges, queries):
    """
    Solve the džumbus problem using tree DP.
    
    dp[node][count][connected] = minimum džumbus needed to have 'count' people
    exchange solutions in the subtree of 'node', where 'connected' indicates
    whether 'node' has exchanged solutions with someone.
    
    For the forest, we add a dummy root with infinite D value.
    """
    if n == 0:
        return [0] * len(queries)
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find connected components and root each tree
    visited = [False] * (n + 1)
    roots = []
    parent = [0] * (n + 1)
    
    # BFS to set up parent relationships
    from collections import deque
    
    for start in range(1, n + 1):
        if not visited[start]:
            roots.append(start)
            queue = deque([start])
            visited[start] = True
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        parent[neighbor] = node
                        queue.append(neighbor)
    
    # Build children list
    children = defaultdict(list)
    for node in range(1, n + 1):
        if parent[node] != 0:
            children[parent[node]].append(node)
    
    # Compute subtree sizes (iterative to avoid recursion limit)
    subtree_size = [0] * (n + 1)
    
    def compute_size_iterative(root):
        # Post-order traversal using stack
        stack = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                subtree_size[node] = 1
                for child in children[node]:
                    subtree_size[node] += subtree_size[child]
            else:
                stack.append((node, True))
                for child in children[node]:
                    stack.append((child, False))
    
    for root in roots:
        compute_size_iterative(root)
    
    # dp[node][count][connected]
    # connected = 0: node hasn't exchanged
    # connected = 1: node has exchanged
    node_dp = {}
    
    def tree_dp_iterative(root):
        """
        Compute DP for entire subtree rooted at root using iterative post-order traversal.
        """
        # Post-order traversal
        stack = [(root, False)]
        order = []
        
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
            else:
                stack.append((node, True))
                for child in children[node]:
                    stack.append((child, False))
        
        # Process nodes in post-order (leaves first)
        for node in order:
            # Base case: leaf node
            if not children[node]:
                node_dp[node] = {(0, 0): 0, (0, 1): INF, (1, 0): INF, (1, 1): INF}
                continue
            
            # Get dp tables for all children (already computed)
            child_dps = [(child, node_dp[child]) for child in children[node]]
            
            # Merge children using prefix DP
            prefix = {(0, 0): 0, (0, 1): INF}
            
            for child, child_dp in child_dps:
                new_prefix = defaultdict(lambda: INF)
                
                for (a, conn_p), cost_p in prefix.items():
                    if cost_p >= INF:
                        continue
                    
                    for (b, conn_c), cost_c in child_dp.items():
                        if cost_c >= INF:
                            continue
                        
                        # Case 1: node doesn't connect with this child
                        if conn_p == 0:
                            child_min = min(child_dp.get((b, 0), INF), child_dp.get((b, 1), INF))
                            if child_min < INF:
                                new_cost = cost_p + child_min
                                new_prefix[(a + b, 0)] = min(new_prefix[(a + b, 0)], new_cost)
                        
                        if conn_p == 1:
                            child_min = min(child_dp.get((b, 0), INF), child_dp.get((b, 1), INF))
                            if child_min < INF:
                                new_cost = cost_p + child_min
                                new_prefix[(a + b, 1)] = min(new_prefix[(a + b, 1)], new_cost)
                        
                        # Case 2: node connects with this child
                        if conn_p == 0 and conn_c == 0:
                            new_cost = cost_p + child_dp.get((b, 0), INF) + d[node] + d[child]
                            new_prefix[(a + b + 2, 1)] = min(new_prefix[(a + b + 2, 1)], new_cost)
                        
                        if conn_p == 0 and conn_c == 1:
                            new_cost = cost_p + child_dp.get((b, 1), INF) + d[node]
                            new_prefix[(a + b + 1, 1)] = min(new_prefix[(a + b + 1, 1)], new_cost)
                        
                        if conn_p == 1 and conn_c == 0:
                            new_cost = cost_p + child_dp.get((b, 0), INF) + d[child]
                            new_prefix[(a + b + 1, 1)] = min(new_prefix[(a + b + 1, 1)], new_cost)
                
                prefix = dict(new_prefix)
            
            node_dp[node] = prefix
    
    # Combine results from all roots (forest)
    # We need to merge the dp tables of all roots
    
    all_results = []
    for root in roots:
        tree_dp_iterative(root)
        root_dp = node_dp[root]
        all_results.append((root, root_dp, subtree_size[root]))
    
    # Merge all trees
    combined = {0: 0}  # combined[count] = min cost
    
    for root, root_dp, size in all_results:
        new_combined = defaultdict(lambda: INF)
        
        for count1, cost1 in combined.items():
            if cost1 == INF:
                continue
            for (count2, connected), cost2 in root_dp.items():
                if cost2 == INF:
                    continue
                # Take min over connected status
                pass
            
            # Get minimum cost for each count in this tree
            tree_min_cost = defaultdict(lambda: INF)
            for (count2, connected), cost2 in root_dp.items():
                tree_min_cost[count2] = min(tree_min_cost[count2], cost2)
            
            for count2, cost2 in tree_min_cost.items():
                new_combined[count1 + count2] = min(new_combined[count1 + count2], cost1 + cost2)
        
        combined = dict(new_combined)
    
    # Sort by count to enable binary search
    sorted_results = sorted(combined.items())
    
    # For each count, find the minimum cost to achieve at least that count
    # Actually we need: for each query S, find max count such that min_cost[count] <= S
    
    # Create a list where result[i] = (cost, count) sorted by cost
    # Then for query S, binary search for largest count with cost <= S
    
    # First, create monotonic: for increasing count, we want non-decreasing cost
    # Actually, we want: counts and their costs, then for query S find max count with cost <= S
    
    # Create pairs (cost, count) and sort
    pairs = [(cost, count) for count, cost in sorted_results if cost < INF]
    pairs.sort()
    
    # Make counts monotonically increasing (keep max count for each cost prefix)
    if not pairs:
        return [0] * len(queries)
    
    # Build (cost, max_count_achievable) list
    processed = []
    max_count = 0
    for cost, count in pairs:
        max_count = max(max_count, count)
        if not processed or processed[-1][1] < max_count:
            processed.append((cost, max_count))
    
    # Answer queries with binary search
    answers = []
    for s in queries:
        # Find largest count where cost <= s
        lo, hi = 0, len(processed)
        while lo < hi:
            mid = (lo + hi) // 2
            if processed[mid][0] <= s:
                lo = mid + 1
            else:
                hi = mid
        
        if lo == 0:
            answers.append(0)
        else:
            answers.append(processed[lo - 1][1])
    
    return answers


def generate_test_case(n, m, q, d_max=10**9, s_max=10**9, tree_type="random", seed=None):
    """
    Generate a test case with given parameters.
    
    Args:
        n: number of friends
        m: number of pairs (edges), must be < n
        q: number of queries
        d_max: maximum value for D_i
        s_max: maximum value for S_i
        tree_type: type of tree structure
        seed: random seed for reproducibility
    
    Returns:
        (input_string, output_string)
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate D values
    d = [0] + [random.randint(1, d_max) for _ in range(n)]  # 1-indexed
    
    # Generate forest
    edges = generate_forest(n, m, tree_type)
    
    # Generate queries
    # To make queries meaningful, we should have some that achieve different counts
    # Calculate total sum to have some queries that can relax everyone
    total_d = sum(d[1:])
    
    queries = []
    # Some queries that might achieve nothing
    queries.extend([random.randint(1, min(s_max, min(d[1:]) - 1 if min(d[1:]) > 1 else 1)) for _ in range(q // 4)])
    # Some queries in medium range
    queries.extend([random.randint(1, min(s_max, total_d // 2 + 1)) for _ in range(q // 4)])
    # Some queries that might achieve a lot
    queries.extend([random.randint(min(s_max, total_d // 2), min(s_max, total_d * 2)) for _ in range(q // 4)])
    # Fill rest randomly
    queries.extend([random.randint(1, s_max) for _ in range(q - len(queries))])
    
    # Solve
    answers = solve(n, d, edges, queries)
    
    # Format input
    input_lines = [f"{n} {m}"]
    input_lines.append(" ".join(map(str, d[1:])))
    for u, v in edges:
        input_lines.append(f"{u} {v}")
    input_lines.append(str(q))
    for s in queries:
        input_lines.append(str(s))
    
    input_string = "\n".join(input_lines) + "\n"
    
    # Format output
    output_string = "\n".join(map(str, answers)) + "\n"
    
    return input_string, output_string


def verify_sample_cases():
    """Verify the solver against the sample cases from the problem."""
    
    # Sample 1
    n, m = 1, 0
    d = [0, 1000]
    edges = []
    queries = [1000]
    answers = solve(n, d, edges, queries)
    assert answers == [0], f"Sample 1 failed: got {answers}, expected [0]"
    print("Sample 1 passed!")
    
    # Sample 2
    n, m = 3, 2
    d = [0, 1, 2, 3]
    edges = [(1, 2), (1, 3)]
    queries = [2, 3, 5]
    answers = solve(n, d, edges, queries)
    assert answers == [0, 2, 2], f"Sample 2 failed: got {answers}, expected [0, 2, 2]"
    print("Sample 2 passed!")
    
    # Sample 3
    n, m = 14, 13
    d = [0, 2, 3, 4, 19, 20, 21, 5, 22, 6, 7, 23, 8, 10, 14]
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (3, 8), (3, 9),
             (4, 10), (8, 11), (10, 13), (10, 12), (12, 14)]
    queries = [45, 44, 23]
    answers = solve(n, d, edges, queries)
    assert answers == [8, 7, 5], f"Sample 3 failed: got {answers}, expected [8, 7, 5]"
    print("Sample 3 passed!")
    
    print("All sample cases verified!")


if __name__ == "__main__":
    # First verify correctness against sample cases
    verify_sample_cases()
    
    # Parameters for test case generation:
    # N = 1000, M = 999 (full tree), Q = 200000
    # This should stress the O(N^2 + Q*log(N)) solution significantly.
    #
    # For a ~10x slowdown comparison:
    # - Small test: N=100, M=99, Q=20000 (baseline)
    # - Large test: N=1000, M=999, Q=200000 (10x more N, 10x more Q)
    #
    # Since tree DP is O(N^2), increasing N by 10x gives ~100x slowdown.
    # Increasing Q by 10x gives ~10x slowdown for query answering.
    # Combined: significant slowdown for the full solution.
    
    # Generate multiple test cases with different characteristics
    # test_configs = [
    #     # (N, M, Q, d_max, s_max, tree_type, description)
    #     (100, 99, 1000, 10**6, 10**6, "chain", "small chain"),
    #     (100, 99, 1000, 10**6, 10**6, "balanced", "small balanced"),
    #     (100, 99, 1000, 10**6, 10**6, "star", "small star"),
    #     (500, 499, 50000, 10**9, 10**9, "random", "medium random"),
    #     (500, 499, 50000, 10**9, 10**9, "chain", "medium chain"),
    #     (1000, 999, 200000, 10**9, 10**9, "random", "large random tree"),
    #     (1000, 999, 200000, 10**9, 10**9, "chain", "large chain (worst DP)"),
    #     (1000, 999, 200000, 10**9, 10**9, "balanced", "large balanced tree"),
    #     (1000, 500, 200000, 10**9, 10**9, "random", "large forest"),
    #     (1000, 0, 200000, 10**9, 10**9, "random", "isolated nodes only"),
    # ]

    test_configs = [(1000, 999, 200000, 10**9, 10**9, "chain", "large chain (worst DP)")] * 5 + \
        [(1000, 999, 200000, 10**9, 10**9, "balanced", "large balanced tree")] * 5
    
    for i, (n, m, q, d_max, s_max, tree_type, desc) in enumerate(test_configs, 1):
        print(f"Generating test case {i}: {desc} (N={n}, M={m}, Q={q})...")
        
        INPUT_FILE = f"gen.{i}.in"
        OUTPUT_FILE = f"gen.{i}.out"
        
        input_str, output_str = generate_test_case(n, m, q, d_max, s_max, tree_type, seed=42+i)
        
        with open(INPUT_FILE, "w") as f:
            f.write(input_str)
        
        with open(OUTPUT_FILE, "w") as f:
            f.write(output_str)
        
        with open("init.yml", "a") as f:
            f.write(f"- {{in: gen.{i}.in, out: gen.{i}.out, points: 1}}\n")
        
        print(f"  Generated {INPUT_FILE} and {OUTPUT_FILE}")
    
    print("\nAll test cases generated successfully!")