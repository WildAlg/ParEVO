/**
 * Problem: Pizza Delivery
 * Approach:
 * 1. The city grid allows horizontal movement anywhere, but vertical movement is restricted to columns 1 and C.
 * 2. This structure forms a "Ladder Graph" with 2*R nodes (portals at first and last columns).
 * 3. We need to find the shortest path cost for a sequence of D deliveries.
 * 4. We precompute All-Pairs Shortest Paths (APSP) on the Ladder Graph.
 *    - Instead of Dijkstra (O(R^2 log R)), we use a specialized DP Sweep algorithm (O(R^2)).
 *    - The sweep performs 4 passes (Down, Up, Down, Up) to propagate distances vertically and horizontally.
 *    - This is significantly faster and cache-friendly.
 * 5. We process the delivery queries in parallel using the precomputed APSP table.
 * 6. Optimized with Fast I/O and flat memory layout.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

const long long INF = 1e18;

// Fast I/O Reader
struct FastIO {
    static const int BUF_SIZE = 1 << 16;
    char buf[BUF_SIZE];
    int pos = 0, len = 0;

    inline char nextChar() {
        if (pos >= len) {
            pos = 0;
            len = fread(buf, 1, BUF_SIZE, stdin);
            if (len == 0) return EOF;
        }
        return buf[pos++];
    }

    inline int readInt() {
        int x = 0;
        char c = nextChar();
        while (c <= ' ') {
            if (c == EOF) return -1;
            c = nextChar();
        }
        bool neg = false;
        if (c == '-') {
            neg = true;
            c = nextChar();
        }
        while (c >= '0' && c <= '9') {
            x = x * 10 + (c - '0');
            c = nextChar();
        }
        return neg ? -x : x;
    }
} io;

int R, C;
vector<int> grid_flat; // Flattened grid
vector<long long> row_prefix; // Flattened prefix sums

// Helper to get intra-row distance: path from c1 to c2
// Cost includes c2, excludes c1.
inline long long get_row_dist(int r, int c1, int c2) {
    if (c1 == c2) return 0;
    int row_off = r * C;
    if (c1 < c2) {
        return row_prefix[row_off + c2] - row_prefix[row_off + c1];
    } else {
        long long val_c2 = (c2 == 0) ? 0 : row_prefix[row_off + c2 - 1];
        return row_prefix[row_off + c1 - 1] - val_c2;
    }
}

int main() {
    // Read Dimensions
    R = io.readInt();
    C = io.readInt();
    if (R == -1) return 0;

    grid_flat.resize(R * C);
    row_prefix.resize(R * C);

    // Read grid and build prefix sums
    for (int i = 0; i < R; i++) {
        long long current_p = 0;
        int row_off = i * C;
        for (int j = 0; j < C; j++) {
            int val = io.readInt();
            grid_flat[row_off + j] = val;
            current_p += val;
            row_prefix[row_off + j] = current_p;
        }
    }

    int num_nodes = 2 * R;
    // Flattened Distance Matrix: apsp[src * num_nodes + dest]
    // Nodes 0..R-1 are Left Column, R..2R-1 are Right Column
    vector<long long> apsp(num_nodes * num_nodes);

    // Compute APSP using optimized 4-pass sweep in parallel
    // This replaces Dijkstra with a linear scan per source, reducing complexity to O(R^2)
    parlay::parallel_for(0, num_nodes, [&](size_t src) {
        long long* d = &apsp[src * num_nodes];
        for(int i = 0; i < num_nodes; ++i) d[i] = INF;
        d[src] = 0;

        // Lambda to relax horizontal edges at row r
        auto relax_horizontal = [&](int r) {
            int u_L = r;
            int u_R = r + R;
            long long dist_L = d[u_L];
            long long dist_R = d[u_R];
            
            if (dist_L == INF && dist_R == INF) return;

            long long w_LR = 0, w_RL = 0;
            if (C > 1) {
                w_LR = row_prefix[r * C + C - 1] - row_prefix[r * C];
                w_RL = row_prefix[r * C + C - 2];
            }
            // If C=1, w_LR = w_RL = 0, effectively syncing d[u_L] and d[u_R]

            // Relax L -> R
            if (dist_L != INF) {
                if (dist_L + w_LR < d[u_R]) {
                    d[u_R] = dist_L + w_LR;
                    dist_R = d[u_R]; // Update local for immediate use
                }
            }
            // Relax R -> L
            if (dist_R != INF) {
                if (dist_R + w_RL < d[u_L]) {
                    d[u_L] = dist_R + w_RL;
                }
            }
        };

        // Perform 2 iterations of Down-Up sweeps to propagate distances fully
        for(int iter = 0; iter < 2; ++iter) {
            // Pass 1: Sweep Down (Propagate reachability downwards)
            for (int r = 0; r < R; ++r) {
                relax_horizontal(r);
                if (r < R - 1) {
                    // Vertical moves to next row
                    if (d[r] != INF) {
                        int w = grid_flat[(r+1)*C];
                        if (d[r] + w < d[r+1]) d[r+1] = d[r] + w;
                    }
                    if (d[r+R] != INF) {
                        int w = grid_flat[(r+1)*C + C - 1];
                        if (d[r+R] + w < d[r+1+R]) d[r+1+R] = d[r+R] + w;
                    }
                }
            }

            // Pass 2: Sweep Up (Propagate reachability upwards)
            for (int r = R - 1; r >= 0; --r) {
                relax_horizontal(r);
                if (r > 0) {
                    // Vertical moves to prev row
                    if (d[r] != INF) {
                        int w = grid_flat[(r-1)*C];
                        if (d[r] + w < d[r-1]) d[r-1] = d[r] + w;
                    }
                    if (d[r+R] != INF) {
                        int w = grid_flat[(r-1)*C + C - 1];
                        if (d[r+R] + w < d[r-1+R]) d[r-1+R] = d[r+R] + w;
                    }
                }
            }
        }
    });

    int D = io.readInt();
    struct Loc { int r, c; };
    vector<Loc> locs(D + 1);
    locs[0] = {0, 0}; // Start at (1,1) -> (0,0)
    for(int i = 0; i < D; ++i) {
        locs[i+1].r = io.readInt() - 1;
        locs[i+1].c = io.readInt() - 1;
    }

    // Process all delivery segments in parallel
    auto costs = parlay::tabulate(D, [&](size_t i) {
        int r1 = locs[i].r;
        int c1 = locs[i].c;
        int r2 = locs[i+1].r;
        int c2 = locs[i+1].c;

        long long min_dist = INF;

        // Option 1: Direct horizontal move (only valid if same row)
        if (r1 == r2) {
            min_dist = min(min_dist, get_row_dist(r1, c1, c2));
        }

        // Option 2: Via portals (Left or Right column)
        long long to_L = get_row_dist(r1, c1, 0);
        long long to_R = get_row_dist(r1, c1, C-1);
        
        long long from_L = get_row_dist(r2, 0, c2);
        long long from_R = get_row_dist(r2, C-1, c2);

        // Distances from portals of row r1
        const long long* d = &apsp[r1 * num_nodes];
        const long long* dR = &apsp[(r1 + R) * num_nodes];

        // Check paths through all 4 portal combinations
        // Start -> Portal1 -> ... -> Portal2 -> End
        if (d[r2] != INF) min_dist = min(min_dist, to_L + d[r2] + from_L);
        if (d[r2+R] != INF) min_dist = min(min_dist, to_L + d[r2+R] + from_R);

        if (dR[r2] != INF) min_dist = min(min_dist, to_R + dR[r2] + from_L);
        if (dR[r2+R] != INF) min_dist = min(min_dist, to_R + dR[r2+R] + from_R);

        return min_dist;
    });

    // Total time is start node weight + sum of segment costs
    long long total_time = grid_flat[0] + parlay::reduce(costs);
    cout << total_time << endl;

    return 0;
}