/**
 * Solution for Zrakoplović
 * 
 * Approach: Optimized Parallel BFS with ParlayLib
 * 
 * Optimizations:
 * 1. Padded Grid (N+2)^3 to avoid boundary checks.
 * 2. Implicit Graph with constant expansion (8 neighbors) to avoid graph construction overhead.
 * 3. Delayed Sequence generation to avoid intermediate memory allocation/writes for candidates.
 * 4. Relaxed atomic loads for visited check to reduce cache contention.
 * 5. Separate grid and distance arrays to avoid false sharing on read-only grid data.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    int xs, ys, zs;
    cin >> xs >> ys >> zs;
    int xe, ye, ze;
    cin >> xe >> ye >> ze;

    // Dimensions with padding (0 and n+1 are boundary walls)
    int dim = n + 2;
    size_t total_nodes = (size_t)dim * dim * dim;

    // Strides for flattened indexing
    // idx = z * stride_z + x * stride_x + y * stride_y
    int stride_y = 1;
    int stride_x = dim;
    int stride_z = dim * dim;

    int start_idx = zs * stride_z + xs * stride_x + ys * stride_y;
    int end_idx = ze * stride_z + xe * stride_x + ye * stride_y;

    if (start_idx == end_idx) {
        cout << 0 << endl;
        return 0;
    }

    // Grid: 0 = free, 1 = blocked (cloud/wall)
    // Initialize with 1 to create padded boundaries
    parlay::sequence<char> grid(total_nodes, 1);

    // Read Input
    // Input coords are 1-based, matching our padded grid 1..N
    for (int z = 1; z <= n; ++z) {
        for (int x = 1; x <= n; ++x) {
            string row;
            cin >> row;
            int base = z * stride_z + x * stride_x;
            for (int y = 1; y <= n; ++y) {
                if (row[y-1] == '0') {
                    grid[base + y] = 0;
                }
            }
        }
    }

    // Distance array, initialized to -1
    // We use a parlay sequence for memory management
    parlay::sequence<int> dist(total_nodes, -1);
    dist[start_idx] = 0;

    // Frontier initialized with start node
    parlay::sequence<int> frontier;
    frontier.push_back(start_idx);

    int current_dist = 0;

    // Offsets for 6 neighbors + 2 dummies (0)
    // Using 8 allows efficient bitwise operations (shift/mask)
    const int offsets[8] = {
        stride_y, -stride_y,
        stride_x, -stride_x,
        stride_z, -stride_z,
        0, 0
    };

    // BFS Loop
    while (!frontier.empty()) {
        // Check if destination reached
        if (dist[end_idx] != -1) {
            cout << dist[end_idx] << endl;
            return 0;
        }

        // Generate candidates lazily using delayed_seq
        // This avoids allocating and writing a large candidates array
        size_t num_candidates = frontier.size() * 8;
        
        auto candidates_delayed = parlay::delayed_seq<int>(num_candidates, [&](size_t i) {
            // i >> 3 is i / 8; i & 7 is i % 8
            int u = frontier[i >> 3];
            int offset = offsets[i & 7];
            return u + offset;
        });

        // Filter valid candidates and atomically mark them visited
        auto next_frontier = parlay::filter(candidates_delayed, [&](int v) {
            // 1. Check grid (Read-Only) - cheap
            if (grid[v] != 0) return false;
            
            // 2. Optimistic Visited Check (Relaxed Read) - avoids cache contention
            // We use __atomic_load_n for a relaxed atomic load
            if (__atomic_load_n(&dist[v], __ATOMIC_RELAXED) != -1) return false;
            
            // 3. Atomic CAS - synchronization point
            // Try to update dist[v] from -1 to current_dist + 1
            // __sync_bool_compare_and_swap acts as a full barrier
            return __sync_bool_compare_and_swap(&dist[v], -1, current_dist + 1);
        });

        frontier = std::move(next_frontier);
        current_dist++;
    }

    // Final check
    if (dist[end_idx] != -1) {
        cout << dist[end_idx] << endl;
    } else {
        cout << -1 << endl;
    }

    return 0;
}