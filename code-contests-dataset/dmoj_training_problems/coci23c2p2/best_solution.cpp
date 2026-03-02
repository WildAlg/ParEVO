/**
 * Solution for Zrakoplović
 * 
 * Approach: Optimized Parallel BFS with ParlayLib
 * 
 * Key Optimizations:
 * 1. Exact Neighbor Expansion:
 *    - Each node expands to exactly 6 neighbors (up, down, left, right, forward, backward).
 *    - We use a `delayed_seq` of size `frontier.size() * 6` and integer arithmetic (`i/6`, `i%6`) 
 *      to generate candidates on the fly.
 *    - This avoids the 25% overhead of dummy candidates present in bitwise-optimized 8-neighbor approaches,
 *      reducing random memory accesses to the `visited` array which is the primary bottleneck.
 * 
 * 2. Compact State Array:
 *    - Use `uint8_t` (1 byte) for the `visited` array to maximize cache efficiency.
 *    - 0: Free and Unvisited
 *    - 1: Blocked (Cloud/Wall) or Visited
 *    - Initializing the array to 1 handles the padded boundaries and clouds uniformly.
 * 
 * 3. Padded Grid:
 *    - The grid is padded to size (N+2)^3, eliminating boundary checks in the inner loop.
 * 
 * 4. Fused Generate-Filter:
 *    - Use `parlay::delayed_seq` + `parlay::filter` to stream candidates and update state 
 *      without intermediate memory allocation for candidates.
 * 
 * 5. Relaxed Atomics:
 *    - Use `__atomic_load_n` and `__atomic_compare_exchange_n` with `__ATOMIC_RELAXED` 
 *      for thread-safe updates with minimal synchronization overhead.
 */

#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <algorithm>
#include <array>
#include <cstdint>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

int main() {
    // Optimize standard I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    int xs, ys, zs;
    cin >> xs >> ys >> zs;
    int xe, ye, ze;
    cin >> xe >> ye >> ze;

    // Dimensions with padding (0 and n+1 are boundary walls)
    // The grid is embedded in a (N+2)x(N+2)x(N+2) cube.
    int dim = n + 2;
    size_t total_nodes = (size_t)dim * dim * dim;

    // Strides for flattened indexing
    // y is the inner-most dimension (stride 1) for best locality with row-major input
    int stride_y = 1;
    int stride_x = dim;
    int stride_z = dim * dim;

    // Helper to calculate flattened index
    auto get_idx = [&](int x, int y, int z) {
        return z * stride_z + x * stride_x + y * stride_y;
    };

    int start_idx = get_idx(xs, ys, zs);
    int end_idx = get_idx(xe, ye, ze);

    // Edge case: Start and end are the same
    if (start_idx == end_idx) {
        cout << 0 << endl;
        return 0;
    }

    // State Array
    // 0: Free and Unvisited
    // 1: Blocked (Cloud/Wall) or Visited
    // Initialize everything to 1 (Blocked) effectively creating the padded walls.
    parlay::sequence<uint8_t> visited(total_nodes, 1);
    uint8_t* __restrict__ visited_ptr = visited.data();

    // Read Input
    // Fill the valid inner region (1 to n)
    // Buffer for reading rows (max N=100)
    char buffer[105];

    for (int z = 1; z <= n; ++z) {
        for (int x = 1; x <= n; ++x) {
            cin >> buffer;
            // Calculate base index for the start of the row
            int base = z * stride_z + x * stride_x;
            for (int y = 1; y <= n; ++y) {
                if (buffer[y-1] == '0') {
                    // Mark as free/unvisited
                    visited_ptr[base + y] = 0;
                }
            }
        }
    }

    // Initialize BFS
    visited[start_idx] = 1; // Mark start as visited
    parlay::sequence<int> frontier;
    frontier.reserve(n * n * 2); // Pre-allocate heuristic size
    frontier.push_back(start_idx);

    int current_dist = 0;

    // Offsets for 6 neighbors
    // Captured by value in lambda for speed
    const std::array<int, 6> offsets = {
        stride_y, -stride_y,
        stride_x, -stride_x,
        stride_z, -stride_z
    };

    // Parallel BFS Loop
    while (!frontier.empty()) {
        const int* __restrict__ frontier_ptr = frontier.data();
        size_t frontier_size = frontier.size();

        // Generate candidates lazily
        // Each node in the frontier generates 6 potential candidates
        // We use integer arithmetic (i/6, i%6) which is fast and avoids checking dummy neighbors
        auto candidates = parlay::delayed_seq<int>(frontier_size * 6, [frontier_ptr, offsets](size_t i) {
            return frontier_ptr[i / 6] + offsets[i % 6];
        });

        // Filter valid candidates and atomically mark visited
        // parlay::filter executes in parallel
        auto next_frontier = parlay::filter(candidates, [visited_ptr](int v) {
            // 1. Optimistic Check:
            // Check if the node is 0 (Unvisited/Free).
            // If it is 1 (Blocked/Visited), this fails fast.
            // Relaxed load is sufficient.
            if (__atomic_load_n(&visited_ptr[v], __ATOMIC_RELAXED)) return false;

            // 2. Atomic Compare-And-Swap (CAS):
            // Try to update visited[v] from 0 to 1.
            // Returns true only if this thread successfully updated the value.
            // This ensures each node is added to the next frontier exactly once.
            uint8_t expected = 0;
            return __atomic_compare_exchange_n(&visited_ptr[v], &expected, 1, 
                                             false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
        });

        frontier = std::move(next_frontier);
        current_dist++;

        // Check if destination was reached in this step
        // If end_idx was added to frontier, visited[end_idx] is now 1.
        // We check this in the main thread which is safe due to Parlay's synchronization.
        if (visited[end_idx]) {
            cout << current_dist << endl;
            return 0;
        }
    }

    // If loop exits, destination is unreachable
    cout << -1 << endl;

    return 0;
}