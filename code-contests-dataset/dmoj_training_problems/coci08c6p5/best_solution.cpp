/**
 * Problem: Pizza Delivery
 * Approach: Fused Compute-Query with Ladder Graph Optimization
 * 1. The city grid allows horizontal movement anywhere, but vertical movement is restricted to columns 1 and C.
 * 2. This structure forms a "Ladder Graph" where each row has two portals: L_r (left) and R_r (right).
 * 3. We precompute "Best Crossing" (BC) costs for each row, representing the minimal cost to travel 
 *    between L_r and R_r (possibly via other rows).
 * 4. We group the D delivery queries by their starting row to optimize memory locality.
 * 5. We process each active starting row in parallel:
 *    - Perform a Single-Source Shortest Path (SSSP) sweep on the ladder graph to compute distances 
 *      from the source row's portals to all other rows.
 *    - We use uninitialized vectors (via empty default constructor) and avoid memset to optimize for small R.
 *    - We use precomputed column arrays to optimize memory access during sweeps.
 *    - Answer all queries starting from this row.
 * 6. This approach is O(R^2 + D) in time and O(R*C + D) in space.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Optimization pragmas
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

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

struct Query {
    int c1;
    int r2, c2;
};

// Struct to keep L and R distances together.
// Default constructor does nothing to avoid initialization overhead.
struct RowDist {
    long long L, R;
    RowDist() {} 
    RowDist(long long l, long long r) : L(l), R(r) {}
};

int R, C;
// Padded prefix sums: row_prefix[r * (C+1) + c] stores sum of weights in row r from column 0 to c-1.
vector<long long> row_prefix; 
vector<int> col0;
vector<int> colLast;

// Helper to get intra-row distance: path from c1 to c2
// Cost includes c2, excludes c1.
// Uses padded prefix sums for cleaner logic.
inline long long get_row_dist(int r, int c1, int c2) {
    if (c1 == c2) return 0;
    int row_off = r * (C + 1);
    // Prefix array P[k] = sum(0..k-1)
    // Range (c1, c2]:
    // If c1 < c2: sum(c1+1..c2) = P[c2+1] - P[c1+1]
    // If c1 > c2: sum(c2..c1-1) = P[c1] - P[c2]
    if (c1 < c2) {
        return row_prefix[row_off + c2 + 1] - row_prefix[row_off + c1 + 1];
    } else {
        return row_prefix[row_off + c1] - row_prefix[row_off + c2];
    }
}

int main() {
    // Read Dimensions
    R = io.readInt();
    C = io.readInt();
    if (R == -1) return 0;

    // Allocate memory
    // C+1 columns for 1-based prefix sums (0-th element is 0)
    row_prefix.resize(R * (C + 1));
    col0.resize(R);
    colLast.resize(R);

    int start_node_cost = 0;

    // Read grid and build prefix sums
    for (int i = 0; i < R; i++) {
        long long current_p = 0;
        int row_off = i * (C + 1);
        row_prefix[row_off] = 0; // P[0] = 0
        for (int j = 0; j < C; j++) {
            int val = io.readInt();
            if (i == 0 && j == 0) start_node_cost = val;
            
            // Store columns for ladder graph
            if (j == 0) col0[i] = val;
            if (j == C - 1) colLast[i] = val;
            
            current_p += val;
            row_prefix[row_off + j + 1] = current_p;
        }
    }

    // Precompute Best Crossing Costs (BC)
    // BC_LR[r]: min cost to go from L_r (r, 0) to R_r (r, C-1)
    // BC_RL[r]: min cost to go from R_r (r, C-1) to L_r (r, 0)
    vector<long long> BC_LR(R), BC_RL(R);

    // Initialize with direct horizontal costs
    for(int r=0; r<R; ++r) {
        BC_LR[r] = get_row_dist(r, 0, C - 1);
        BC_RL[r] = get_row_dist(r, C - 1, 0);
    }

    // Sweep Down for BC
    for(int r=0; r < R - 1; ++r) {
        long long cost_via_up = (long long)col0[r] + BC_LR[r] + colLast[r+1];
        if (cost_via_up < BC_LR[r+1]) BC_LR[r+1] = cost_via_up;

        long long cost_via_up_RL = (long long)colLast[r] + BC_RL[r] + col0[r+1];
        if (cost_via_up_RL < BC_RL[r+1]) BC_RL[r+1] = cost_via_up_RL;
    }

    // Sweep Up for BC
    for(int r=R-1; r > 0; --r) {
        long long cost_via_down = (long long)col0[r] + BC_LR[r] + colLast[r-1];
        if (cost_via_down < BC_LR[r-1]) BC_LR[r-1] = cost_via_down;

        long long cost_via_down_RL = (long long)colLast[r] + BC_RL[r] + col0[r-1];
        if (cost_via_down_RL < BC_RL[r-1]) BC_RL[r-1] = cost_via_down_RL;
    }

    int D = io.readInt();
    
    // Efficiently bucket queries by starting row r1
    vector<int> q_count(R, 0);
    vector<Query> raw_queries(D);
    
    int curr_r = 0, curr_c = 0; // Start at (1,1) -> (0,0)
    for(int i = 0; i < D; ++i) {
        int next_r = io.readInt() - 1;
        int next_c = io.readInt() - 1;
        raw_queries[i] = {curr_c, next_r, next_c};
        q_count[curr_r]++;
        curr_r = next_r;
        curr_c = next_c;
    }

    // Compute start indices for each row bucket
    vector<int> starts(R + 1, 0);
    for(int i=0; i<R; ++i) starts[i+1] = starts[i] + q_count[i];
    
    // Second pass: place queries into buckets
    vector<Query> sorted_queries(D);
    vector<int> current_pos = starts; 
    
    curr_r = 0; 
    for(int i=0; i<D; ++i) {
        int r1 = curr_r;
        sorted_queries[current_pos[r1]++] = raw_queries[i];
        curr_r = raw_queries[i].r2;
    }

    // Process each row in parallel
    // We compute SSSP for each active source row and answer its queries
    auto row_sums = parlay::tabulate(R, [&](size_t r) {
        int start_idx = starts[r];
        int end_idx = starts[r+1];
        if (start_idx == end_idx) return 0LL;

        // Use uninitialized vectors to avoid overhead
        vector<RowDist> dL(R);
        vector<RowDist> dR(R);

        // --- Compute distances from L_r ---
        dL[r].L = 0;
        dL[r].R = BC_LR[r];

        // Sweep Up from r
        for (int i = r - 1; i >= 0; --i) {
            int next = i + 1; // actually the row below in loop direction
            long long w_next_L = col0[i];
            long long w_next_R = colLast[i];
            
            // Propagate from row below (next)
            long long valL = dL[next].L + w_next_L;
            long long valR = dL[next].R + w_next_R;
            
            // Relax Horizontal at current
            long long crossL = valR + BC_RL[i];
            long long crossR = valL + BC_LR[i];
            
            dL[i].L = min(valL, crossL);
            dL[i].R = min(valR, crossR);
        }

        // Sweep Down from r
        for (int i = r + 1; i < R; ++i) {
            int prev = i - 1; // actually the row above in loop direction
            long long w_prev_L = col0[i];
            long long w_prev_R = colLast[i];
            
            // Propagate from row above (prev)
            long long valL = dL[prev].L + w_prev_L;
            long long valR = dL[prev].R + w_prev_R;
            
            // Relax Horizontal at current
            long long crossL = valR + BC_RL[i];
            long long crossR = valL + BC_LR[i];
            
            dL[i].L = min(valL, crossL);
            dL[i].R = min(valR, crossR);
        }

        // --- Compute distances from R_r ---
        dR[r].R = 0;
        dR[r].L = BC_RL[r];

        // Sweep Up from r
        for (int i = r - 1; i >= 0; --i) {
            int next = i + 1;
            long long w_next_L = col0[i];
            long long w_next_R = colLast[i];
            
            long long valL = dR[next].L + w_next_L;
            long long valR = dR[next].R + w_next_R;
            
            long long crossL = valR + BC_RL[i];
            long long crossR = valL + BC_LR[i];
            
            dR[i].L = min(valL, crossL);
            dR[i].R = min(valR, crossR);
        }

        // Sweep Down from r
        for (int i = r + 1; i < R; ++i) {
            int prev = i - 1;
            long long w_prev_L = col0[i];
            long long w_prev_R = colLast[i];
            
            long long valL = dR[prev].L + w_prev_L;
            long long valR = dR[prev].R + w_prev_R;
            
            long long crossL = valR + BC_RL[i];
            long long crossR = valL + BC_LR[i];
            
            dR[i].L = min(valL, crossL);
            dR[i].R = min(valR, crossR);
        }

        long long sum = 0;
        for (int k = start_idx; k < end_idx; ++k) {
            const auto& q = sorted_queries[k];
            long long min_dist = INF;
            
            // Direct path if same row
            if (r == q.r2) {
                min_dist = get_row_dist(r, q.c1, q.c2);
            }
            
            long long to_L = get_row_dist(r, q.c1, 0);
            long long to_R = get_row_dist(r, q.c1, C - 1);

            long long from_L = get_row_dist(q.r2, 0, q.c2);
            long long from_R = get_row_dist(q.r2, C - 1, q.c2);

            // Path via L_r
            long long dL_opt = min(to_L + dL[q.r2].L + from_L, to_L + dL[q.r2].R + from_R);
            // Path via R_r
            long long dR_opt = min(to_R + dR[q.r2].L + from_L, to_R + dR[q.r2].R + from_R);
            
            min_dist = min({min_dist, dL_opt, dR_opt});

            sum += min_dist;
        }
        return sum;
    });

    long long total_time = start_node_cost + parlay::reduce(row_sums);
    cout << total_time << endl;

    return 0;
}