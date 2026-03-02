#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <algorithm>
#include <utility>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/parallel.h>

// Problem Analysis:
// The city grid can be modeled as a directed graph where each cell is a node.
// Each cell has exactly one outgoing edge pointing to an adjacent cell, as specified
// by the character ('N', 'S', 'E', 'W'). This type of graph is known as a functional graph.
// A cat starting in any cell will eventually enter a cycle and stay within its
// weakly connected component. A single trap placed anywhere in a weakly connected component
// is guaranteed to eventually catch all cats within that component.
// Therefore, the minimum number of traps required is equal to the number of
// weakly connected components in this functional graph.
//
// Solution Approach:
// We can find the number of connected components using a Disjoint Set Union (DSU)
// data structure, also known as Union-Find.
// 1. We treat each of the n*m cells as an individual element in the DSU.
// 2. For each cell `i`, we perform a union operation between `i` and the cell it points to.
//    This process is done in parallel for all cells using the parlay library.
// 3. After all union operations are complete, the number of disjoint sets remaining
//    is the number of connected components, which is our answer.
//
// Implementation Details:
// A lock-free parallel DSU is implemented using `std::atomic<int>` for the parent array.
// Path splitting in `find_set` and union-by-index in `unite_sets` are used for efficiency.
// Relaxed memory ordering is used for atomic operations to maximize performance, which is
// correct for this lock-free algorithm as it relies on retry loops rather than strict ordering.

// Finds the representative of the set containing `i` using path splitting.
// Path splitting is a fast, single-pass path compression technique that is
// well-suited for parallel execution. While traversing to the root, it makes
// each node attempt to point to its grandparent.
int find_set(int i, parlay::sequence<std::atomic<int>>& parent) {
    while (i != parent[i].load(std::memory_order_relaxed)) {
        int p = parent[i].load(std::memory_order_relaxed);
        int gp = parent[p].load(std::memory_order_relaxed);
        // Atomically try to set i's parent to its grandparent.
        // This may fail if another thread interferes, which is fine; the path still shortens.
        parent[i].compare_exchange_strong(p, gp, std::memory_order_relaxed, std::memory_order_relaxed);
        // Move up the tree. Reading parent[i] again might allow a bigger jump
        // if the CAS succeeded or another thread updated it.
        i = parent[i].load(std::memory_order_relaxed);
    }
    return i;
}

// Unites the sets containing elements `i` and `j`.
// Uses a retry loop to handle concurrent modifications by other threads.
void unite_sets(int i, int j, parlay::sequence<std::atomic<int>>& parent) {
    while (true) {
        int root_i = find_set(i, parent);
        int root_j = find_set(j, parent);
        
        if (root_i == root_j) {
            return; // Already in the same set.
        }
        
        // Use union-by-index to provide a consistent ordering and prevent cycles
        // in the DSU forest. The root with the smaller index points to the larger one.
        if (root_i > root_j) {
            std::swap(root_i, root_j);
        }
        
        int expected_parent_of_root_i = root_i;
        // Atomically link the smaller root to the larger one.
        if (parent[root_i].compare_exchange_strong(expected_parent_of_root_i, root_j, std::memory_order_relaxed, std::memory_order_relaxed)) {
            break; // Union successful.
        }
        // If CAS fails, another thread interfered; retry the entire operation.
    }
}

int main() {
    // Fast I/O for performance.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int n, m;
    std::cin >> n >> m;

    std::vector<std::string> grid(n);
    for (int i = 0; i < n; ++i) {
        std::cin >> grid[i];
    }

    int num_cells = n * m;
    if (num_cells == 0) {
        std::cout << 0 << std::endl;
        return 0;
    }

    // `next_node[i]` stores the linear index of the cell that cell `i` points to.
    // Created in parallel using parlay::tabulate.
    auto next_node = parlay::tabulate(num_cells, [&](int i) {
        int r = i / m;
        int c = i % m;
        switch (grid[r][c]) {
            case 'N': r--; break;
            case 'S': r++; break;
            case 'E': c++; break;
            case 'W': c--; break;
        }
        return r * m + c;
    });

    // The `parent` array for the DSU, using `std::atomic<int>` for thread-safety.
    parlay::sequence<std::atomic<int>> parent(num_cells);
    
    // Initialize DSU: each cell is its own parent.
    parlay::parallel_for(0, num_cells, [&](int i) {
        parent[i].store(i, std::memory_order_relaxed);
    });

    // In parallel, unite each cell's set with its successor's set.
    parlay::parallel_for(0, num_cells, [&](int i) {
        unite_sets(i, next_node[i], parent);
    });

    // The number of traps is the number of connected components, which is
    // the number of roots in the DSU forest (nodes where parent[i] == i).
    long num_traps = parlay::count_if(parlay::iota<int>(num_cells), [&](int i) {
        return parent[i].load(std::memory_order_relaxed) == i;
    });
    
    std::cout << num_traps << std::endl;

    return 0;
}