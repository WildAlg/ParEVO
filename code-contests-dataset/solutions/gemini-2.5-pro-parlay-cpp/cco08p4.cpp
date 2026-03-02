#include <iostream>
#include <string>
#include <atomic>
#include <algorithm>
#include <iterator>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/parallel.h>

// A high-performance concurrent Union-Find data structure.
// This implementation is tailored for massive parallelism by employing
// different strategies for the unioning phase and the final counting phase.
// It uses path halving for unions to reduce contention and full path compression
// for the final count to ensure accuracy.
struct UnionFind {
    parlay::sequence<std::atomic<int>> parent;
    const int num_elements;

    explicit UnionFind(int size) : num_elements(size) {
        parent = parlay::sequence<std::atomic<int>>(num_elements);
        // Initialize each element to be its own parent in parallel.
        parlay::parallel_for(0, num_elements, [&](int i) {
            parent[i].store(i, std::memory_order_relaxed);
        });
    }

    // A fast `find` operation using path halving.
    // This is a one-pass method that makes each node point to its grandparent.
    // It provides a good balance of path compression and low overhead/contention,
    // making it ideal for the highly concurrent union phase.
    int find_halving(int i) {
        while (i != parent[i].load(std::memory_order_relaxed)) {
            int p = parent[i].load(std::memory_order_relaxed);
            int gp = parent[p].load(std::memory_order_relaxed);
            // Attempt to set parent to grandparent. Failure is non-critical.
            parent[i].compare_exchange_strong(p, gp, std::memory_order_relaxed);
            // Move up the tree. Must read again as another thread might have updated it.
            i = parent[i].load(std::memory_order_relaxed);
        }
        return i;
    }

    // A thorough `find` operation using two-pass full path compression.
    // This is used to completely flatten the trees before the final count,
    // ensuring every node points directly to its root for an accurate result.
    // It's more expensive and thus not used during the main unioning loop.
    void find_full_compress(int i) {
        int root = i;
        // First pass: find the ultimate root.
        while (root != parent[root].load(std::memory_order_relaxed)) {
            root = parent[root].load(std::memory_order_relaxed);
        }
        // Second pass: make all nodes on the path point directly to the root.
        int curr = i;
        while (curr != root) {
            int next = parent[curr].load(std::memory_order_relaxed);
            parent[curr].compare_exchange_strong(next, root, std::memory_order_relaxed);
            curr = next;
        }
    }

    // Merges the sets containing elements `i` and `j`.
    // Uses `find_halving` for speed and a CAS loop to handle concurrent updates.
    void unite(int i, int j) {
        while (true) {
            int root_i = find_halving(i);
            int root_j = find_halving(j);
            if (root_i == root_j) {
                return; // Already in the same set.
            }
            
            // A simple, deterministic linking strategy: union-by-index.
            // Always attach the tree with the larger root index to the smaller one.
            if (root_i < root_j) {
                std::swap(root_i, root_j);
            }
            
            int root_i_parent = root_i;
            // Atomically set the parent of the larger root to the smaller root.
            if (parent[root_i].compare_exchange_strong(root_i_parent, root_j, std::memory_order_relaxed)) {
                return; // Union successful.
            }
            // CAS failed, indicating a concurrent update. Retry the whole operation.
        }
    }

    // Counts the total number of disjoint sets.
    long count_sets() {
        // First, run full path compression on all elements in parallel.
        // This flattens the data structure, making counting trivial.
        parlay::parallel_for(0, num_elements, [&](int i) {
            find_full_compress(i);
        });
        
        // Count the number of roots. A node `i` is a root if `parent[i] == i`.
        // This is done in parallel using a reduction over a delayed sequence.
        auto is_root = parlay::delayed_seq<long>(num_elements, [&](size_t i) {
            return (parent[i].load(std::memory_order_relaxed) == (int)i) ? 1L : 0L;
        });
        return parlay::reduce(is_root);
    }
};

int main() {
    // Fast I/O
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int n, m;
    std::cin >> n >> m;

    int total_cells = n * m;
    if (total_cells == 0) {
        std::cout << 0 << std::endl;
        return 0;
    }
    
    // Read the grid into a flat sequence for better cache performance and simpler indexing.
    parlay::sequence<char> grid(total_cells);
    for (int i = 0; i < n; ++i) {
        std::string row;
        std::cin >> row;
        // std::copy is efficient for moving the row data.
        std::copy(row.begin(), row.end(), grid.begin() + i * m);
    }

    UnionFind uf(total_cells);

    // The core computation: build the graph and find connected components simultaneously.
    // This is done in a single, fine-grained parallel loop over all cells, which
    // provides good load balancing for the Parlay scheduler.
    // It uses efficient direct index arithmetic instead of division/modulo.
    parlay::parallel_for(0, total_cells, [&](int current_idx) {
        int next_idx;
        switch (grid[current_idx]) {
            case 'N': next_idx = current_idx - m; break;
            case 'S': next_idx = current_idx + m; break;
            case 'W': next_idx = current_idx - 1; break;
            case 'E': next_idx = current_idx + 1; break;
            default: next_idx = current_idx; // Should not occur given problem constraints
        }
        uf.unite(current_idx, next_idx);
    });

    // The minimum number of traps is equal to the number of connected components.
    std::cout << uf.count_sets() << std::endl;

    return 0;
}