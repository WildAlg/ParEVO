#include <iostream>
#include <string>
#include <atomic>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

// This program solves the shortest path problem on a 3D grid using a parallel Breadth-First Search (BFS).
// The Parlay library is used to parallelize the BFS exploration, which is efficient for this problem size.

int main() {
    // Use fast I/O for performance, a standard practice in competitive programming.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // --- 1. Read Input and Problem Setup ---
    int n;
    std::cin >> n;

    int xs, ys, zs;
    std::cin >> xs >> ys >> zs;

    int xe, ye, ze;
    std::cin >> xe >> ye >> ze;

    // A lambda function to convert 1-based 3D coordinates to a 0-based 1D index.
    // This allows representing the 3D grid as a flat 1D array, which is memory-efficient
    // and simplifies neighbor calculations using index arithmetic.
    auto to_1d = [n](int x, int y, int z) {
        return (z - 1) * n * n + (x - 1) * n + (y - 1);
    };

    const int start_idx = to_1d(xs, ys, zs);
    const int end_idx = to_1d(xe, ye, ze);

    // Trivial case: If the start and end positions are the same, no moves are needed.
    if (start_idx == end_idx) {
        std::cout << 0 << std::endl;
        return 0;
    }

    // Read the grid configuration. `true` represents a cloud (obstacle), `false` is empty space.
    // Using parlay::sequence<bool> is memory-efficient.
    const int total_cubes = n * n * n;
    parlay::sequence<bool> grid_is_cloud(total_cubes);
    for (int z = 1; z <= n; ++z) {
        for (int x = 1; x <= n; ++x) {
            std::string row;
            std::cin >> row;
            for (int y = 1; y <= n; ++y) {
                grid_is_cloud[to_1d(x, y, z)] = (row[y - 1] == '1');
            }
        }
    }
    
    // --- 2. Parallel BFS Initialization ---
    
    // `dist` array stores the shortest distance from the start. -1 indicates an unvisited cube.
    // `std::atomic` is crucial for thread-safe updates in the parallel BFS.
    // Relaxed memory ordering is used for performance, as strict ordering is not required for correctness here.
    parlay::sequence<std::atomic<int>> dist(total_cubes);
    parlay::parallel_for(0, total_cubes, [&](size_t i) {
        dist[i].store(-1, std::memory_order_relaxed);
    });

    // The `frontier` holds the set of nodes to be visited at the current level of the BFS.
    // It is initialized with the starting node.
    parlay::sequence<int> frontier;
    dist[start_idx].store(0, std::memory_order_relaxed);
    frontier.push_back(start_idx);

    int current_distance = 0;
    const int n_squared = n * n; // Pre-calculate n*n for efficiency inside the loop.

    // --- 3. Parallel BFS Main Loop ---
    while (!frontier.empty()) {
        // Optimization: If the destination has been reached, we can terminate early.
        if (dist[end_idx].load(std::memory_order_relaxed) != -1) {
            break;
        }
        
        current_distance++;
        
        // In parallel, generate the next frontier by exploring neighbors of all nodes in the current frontier.
        auto next_frontier = parlay::flatten(parlay::map(frontier, [&](int u) {
            parlay::sequence<int> neighbors_to_add;
            
            // Deconstruct the 1D index `u` into 0-indexed coordinates for efficient boundary checking.
            const int z0 = u / n_squared;
            const int x0 = (u % n_squared) / n;
            const int y0 = u % n;

            // Helper lambda to check a potential neighbor.
            auto check_and_add = [&](int neighbor_idx) {
                // Check if the neighbor is not a cloud.
                if (!grid_is_cloud[neighbor_idx]) {
                    int expected = -1;
                    // Atomically check if unvisited and update distance. This compare-and-swap (CAS)
                    // is the core of the thread-safe parallel BFS. If it succeeds, this thread
                    // is the first to discover this node at this distance level.
                    if (dist[neighbor_idx].compare_exchange_strong(expected, current_distance, std::memory_order_relaxed, std::memory_order_relaxed)) {
                        neighbors_to_add.push_back(neighbor_idx);
                    }
                }
            };

            // Explore the 6 axis-aligned neighbors using direct 1D index manipulation.
            // Boundary checks use the pre-calculated 0-indexed coordinates.
            if (y0 > 0)     check_and_add(u - 1);
            if (y0 < n - 1) check_and_add(u + 1);
            if (x0 > 0)     check_and_add(u - n);
            if (x0 < n - 1) check_and_add(u + n);
            if (z0 > 0)     check_and_add(u - n_squared);
            if (z0 < n - 1) check_and_add(u + n_squared);

            return neighbors_to_add;
        }));

        // The newly discovered nodes become the frontier for the next iteration.
        frontier = std::move(next_frontier);
    }

    // --- 4. Output Result ---
    // The final distance to the end node. If it's unreachable, dist will remain -1.
    std::cout << dist[end_idx].load(std::memory_order_relaxed) << std::endl;

    return 0;
}