#include <iostream>
#include <string>
#include <atomic>
#include <tuple>

// Include necessary parlay headers
#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

// Helper function to convert 3D coordinates (0-indexed) to a 1D index.
// The 1D array is laid out in z-major, then x-major, then y-major order.
inline size_t get_index(int x, int y, int z, int n) {
    return static_cast<size_t>(z) * n * n + static_cast<size_t>(x) * n + static_cast<size_t>(y);
}

// Helper function to convert a 1D index back to 3D coordinates (0-indexed).
inline std::tuple<int, int, int> get_coords(size_t index, int n) {
    const size_t n_squared = static_cast<size_t>(n) * n;
    int z = index / n_squared;
    size_t rem = index % n_squared;
    int x = rem / n;
    int y = rem % n;
    return {x, y, z};
}

int main() {
    // Use fast I/O
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int n;
    std::cin >> n;

    int xs, ys, zs;
    std::cin >> xs >> ys >> zs;
    // Convert from 1-indexed to 0-indexed for internal use
    xs--; ys--; zs--;

    int xe, ye, ze;
    std::cin >> xe >> ye >> ze;
    // Convert from 1-indexed to 0-indexed
    xe--; ye--; ze--;

    size_t total_cells = static_cast<size_t>(n) * n * n;
    // Store the grid as a flat sequence of characters ('0' for empty, '1' for cloud)
    parlay::sequence<char> grid(total_cells);

    // Read the grid data. The input is given as n matrices for heights 1 to n.
    // Each matrix is n x n, where rows correspond to x and columns to y.
    for (int z = 0; z < n; ++z) {
        for (int x = 0; x < n; ++x) {
            std::string row;
            std::cin >> row;
            for (int y = 0; y < n; ++y) {
                grid[get_index(x, y, z, n)] = row[y];
            }
        }
    }

    size_t start_idx = get_index(xs, ys, zs, n);
    size_t end_idx = get_index(xe, ye, ze, n);

    // If start and end positions are the same, the distance is 0 wing flaps.
    if (start_idx == end_idx) {
        std::cout << 0 << std::endl;
        return 0;
    }

    // Distance array, initialized to -1 (representing unvisited).
    // We use std::atomic<int> to allow for safe parallel updates from multiple threads.
    parlay::sequence<std::atomic<int>> dist(total_cells);
    parlay::parallel_for(0, total_cells, [&](size_t i) {
        dist[i].store(-1);
    });

    // The frontier for the parallel Breadth-First Search (BFS). It holds nodes to visit at the current level.
    parlay::sequence<size_t> frontier;
    frontier.push_back(start_idx);
    
    dist[start_idx].store(0);
    int current_dist = 0;

    // Main parallel BFS loop. Continues as long as there are nodes in the frontier.
    while (!frontier.empty()) {
        // Optimization: check if the destination has been reached.
        // This is an O(1) atomic read, which is faster than scanning the frontier.
        if (dist[end_idx].load() != -1) {
            break;
        }

        // Increment distance for the next level of the BFS.
        current_dist++;

        // For each node in the current frontier, find its unvisited neighbors in parallel.
        auto nested_neighbors = parlay::map(frontier, [&](size_t u) {
            auto [x, y, z] = get_coords(u, n);
            parlay::sequence<size_t> new_neighbors;
            new_neighbors.reserve(6); // Pre-allocate memory for up to 6 neighbors

            // Define the 6 possible moves in 3D space (axis-parallel).
            const int dx[] = {1, -1, 0, 0, 0, 0};
            const int dy[] = {0, 0, 1, -1, 0, 0};
            const int dz[] = {0, 0, 0, 0, 1, -1};

            for (int i = 0; i < 6; ++i) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                int nz = z + dz[i];

                // Check if the neighbor is within the grid boundaries.
                if (nx >= 0 && nx < n && ny >= 0 && ny < n && nz >= 0 && nz < n) {
                    size_t v_idx = get_index(nx, ny, nz, n);
                    // Check if the neighbor cell is not a cloud.
                    if (grid[v_idx] == '0') {
                        int expected = -1;
                        // Atomically try to mark the neighbor as visited with the new distance.
                        if (dist[v_idx].compare_exchange_strong(expected, current_dist)) {
                            new_neighbors.push_back(v_idx);
                        }
                    }
                }
            }
            return new_neighbors;
        });
        
        // Flatten the sequence of neighbor sequences into a single new frontier for the next BFS level.
        frontier = parlay::flatten(nested_neighbors);
    }

    // If the loop finishes, dist[end_idx] holds the shortest distance.
    // If it's still -1, the destination is unreachable.
    std::cout << dist[end_idx].load() << std::endl;

    return 0;
}