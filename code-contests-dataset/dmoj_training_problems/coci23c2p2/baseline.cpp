/**
 * Solution for Zrakoplović
 * 
 * Problem: Shortest path in a 3D grid with obstacles.
 * Algorithm: Parallel Breadth-First Search (BFS) using the ParlayLib library.
 * 
 * Coordinate Mapping:
 * - x corresponds to the row index (0 to n-1)
 * - y corresponds to the column index (0 to n-1)
 * - z corresponds to the layer/height index (0 to n-1)
 * 
 * Flattened Index: idx = z * n^2 + x * n + y
 * 
 * Complexity:
 * - Time: O(N^3) work, O(N) span (parallel)
 * - Space: O(N^3)
 */

#include <iostream>
#include <vector>
#include <string>
#include <atomic>

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

    // Convert 1-based coordinates to 0-based
    xs--; ys--; zs--;
    xe--; ye--; ze--;

    // Calculate start and end flattened indices
    int start_idx = zs * n * n + xs * n + ys;
    int end_idx = ze * n * n + xe * n + ye;

    // Flattened grid: z * n * n + x * n + y
    // x corresponds to row, y corresponds to column
    // grid[i] is true if there is a cloud (blocked), false otherwise
    vector<bool> grid(n * n * n);

    for (int z = 0; z < n; ++z) {
        for (int x = 0; x < n; ++x) {
            string row;
            cin >> row;
            for (int y = 0; y < n; ++y) {
                int idx = z * n * n + x * n + y;
                grid[idx] = (row[y] == '1');
            }
        }
    }

    // Distance array using atomic integers for thread-safe updates
    // Initialize with -1 (unvisited)
    // We use a raw pointer because vector<atomic> is not copyable/movable
    std::atomic<int>* dist = new std::atomic<int>[n * n * n];
    
    // Parallel initialization of the distance array
    parlay::parallel_for(0, n * n * n, [&](size_t i) {
        dist[i].store(-1, std::memory_order_relaxed);
    });

    // Set distance for start position
    dist[start_idx].store(0, std::memory_order_relaxed);

    // Frontier contains the indices of nodes to process in the current step
    parlay::sequence<int> frontier;
    frontier.push_back(start_idx);

    int current_dist = 0;
    bool found = false;
    int result = -1;

    // BFS Loop
    while (!frontier.empty()) {
        // Check if destination has been reached
        // We check dist[end_idx] because it might have been set in the previous iteration
        int d_end = dist[end_idx].load(std::memory_order_relaxed);
        if (d_end != -1) {
            result = d_end;
            found = true;
            break;
        }

        // Parallel expansion: Generate next level candidates
        auto neighbors_nested = parlay::map(frontier, [&](int u) {
            // Decode index to coordinates
            int temp = u;
            int uy = temp % n;
            temp /= n;
            int ux = temp % n;
            int uz = temp / n;

            int cands[6];
            int count = 0;

            // Generate potential neighbors (up, down, left, right, forward, backward)
            // z-axis
            if (uz > 0) cands[count++] = u - n * n;
            if (uz < n - 1) cands[count++] = u + n * n;
            // x-axis
            if (ux > 0) cands[count++] = u - n;
            if (ux < n - 1) cands[count++] = u + n;
            // y-axis
            if (uy > 0) cands[count++] = u - 1;
            if (uy < n - 1) cands[count++] = u + 1;

            int valid_cands[6];
            int valid_count = 0;
            for (int i = 0; i < count; ++i) {
                int v = cands[i];
                if (!grid[v]) { // Must not be a cloud
                    // Optimistic check to reduce atomic contention:
                    // Only consider if it looks unvisited
                    if (dist[v].load(std::memory_order_relaxed) == -1) {
                        valid_cands[valid_count++] = v;
                    }
                }
            }
            return parlay::sequence<int>(valid_cands, valid_cands + valid_count);
        });

        // Flatten nested sequences to get all candidates
        auto candidates = parlay::flatten(neighbors_nested);
        
        if (candidates.empty()) break;

        // Filter candidates and update distances atomically
        // parlay::filter keeps elements for which the predicate returns true
        auto next_frontier = parlay::filter(candidates, [&](int v) {
            int expected = -1;
            // Atomic Compare-And-Swap: if dist[v] is -1, set to current_dist + 1 and return true
            // This ensures each node is added to the next frontier exactly once
            return dist[v].compare_exchange_strong(expected, current_dist + 1, std::memory_order_relaxed);
        });

        frontier = next_frontier;
        current_dist++;
    }

    // Final check if loop exited without breaking
    if (!found) {
        int d_end = dist[end_idx].load(std::memory_order_relaxed);
        if (d_end != -1) result = d_end;
    }

    cout << result << endl;

    delete[] dist;
    return 0;
}