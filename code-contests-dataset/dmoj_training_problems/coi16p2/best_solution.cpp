/**
 * Solution for Meadow Complexity
 * 
 * Algorithm: Divide and Conquer on rows with Parallelism and Run Compression.
 * 
 * We recursively divide the row range [L, R] into [L, mid] and [mid+1, R].
 * The complexity of a combined meadow P^a_b is calculated by merging components
 * from the left part [a, mid] and right part [mid+1, b].
 * 
 * Key Optimizations:
 * 1. Bitset Grid: Rows are stored as uint64_t for fast bitwise operations.
 * 2. Run Compression: Rows with identical boundary connectivity are grouped.
 *    Since the boundary partition can change at most M times, we have at most M runs.
 * 3. Signature Compression: When merging runs from left and right halves, we only
 *    consider the "active" columns (where both sides have grass). We compress
 *    runs based on their connectivity signature on these active columns.
 * 4. Parallelism: Uses parlay::par_do for recursion and parallel run generation.
 * 5. Memory Efficiency: Uses stack-allocated fixed-size arrays.
 * 6. Optimized Inner Loops: Removed unnecessary memsets and used fast bit manipulation.
 */

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

using namespace std;

// Constants
const int MAX_M = 55;
const int MAX_RUNS = 60; // Bounded by M + small buffer

// Global Data
int N, M;
vector<uint64_t> bit_grid;

// Run structure representing a group of rows with same boundary connectivity
struct Run {
    int8_t partition[MAX_M]; // Component ID for each column at the boundary
    int8_t num_ids;          // Number of distinct component IDs
    long long count;         // Number of rows in this run
    long long sum_comps;     // Sum of complexities for rows in this run
};

// Compressed Run structure for optimized merging
struct CompactRun {
    int8_t sig[MAX_M];       // Canonical partition restricted to active columns
    int8_t num_unique;       // Number of unique IDs in the signature
    long long total_count;   // Sum of counts of aggregated runs
    
    // Sorting for grouping identical signatures
    bool operator<(const CompactRun& other) const {
        return std::memcmp(sig, other.sig, M) < 0;
    }
};

// Optimized DSU with small fixed size
struct TinyDSU {
    int8_t parent[128]; // Sufficient for 2*MAX_M active components
    
    // Initialize only the needed range
    inline void init(int n) {
        for(int i=0; i<n; ++i) parent[i] = (int8_t)i;
    }
    
    inline int find(int i) {
        int root = i;
        while(parent[root] != root) root = parent[root];
        int curr = i;
        while(curr != root) {
            int next = parent[curr];
            parent[curr] = (int8_t)root;
            curr = next;
        }
        return root;
    }
    
    inline bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = (int8_t)root_j;
            return true;
        }
        return false;
    }
};

// Helper to count connected segments in a single row
inline int count_segments(uint64_t row) {
    if (row == 0) return 0;
    return __builtin_popcountll(row & ~(row << 1));
}

// Generate runs for the left side (expanding from mid down to L)
int generate_left_runs(int L, int mid, Run* runs) {
    int8_t b_ids[MAX_M];      // Boundary row IDs (row mid)
    int8_t c_ids[MAX_M];      // Current row IDs
    int8_t next_row_ids[MAX_M]; // Next row IDs (moving away from mid)
    int8_t mapping[128];      // For canonicalization
    TinyDSU dsu;

    uint64_t row_mid = bit_grid[mid-1];
    int next_id = 0;
    int current_V = 0;

    // Initialize for row mid
    memset(b_ids, -1, M);
    uint64_t temp = row_mid;
    while(temp) {
        int j = __builtin_ctzll(temp);
        if (j > 0 && ((row_mid >> (j-1)) & 1)) {
            b_ids[j] = b_ids[j-1];
        } else {
            b_ids[j] = (int8_t)next_id++;
            current_V++;
        }
        temp &= (temp - 1);
    }
    memcpy(c_ids, b_ids, M);

    int n_runs = 0;
    memcpy(runs[0].partition, b_ids, M);
    runs[0].num_ids = (int8_t)next_id;
    runs[0].count = 1;
    runs[0].sum_comps = current_V;
    n_runs = 1;

    for (int i = mid - 1; i >= L; --i) {
        uint64_t row_curr = bit_grid[i-1];
        uint64_t row_prev = bit_grid[i]; // row i+1

        int start_node = next_id;
        int added_nodes = 0;
        int delta_V = 0;

        // Assign IDs to segments in current row. No memset needed as we only access valid bits.
        temp = row_curr;
        while(temp) {
            int j = __builtin_ctzll(temp);
            if (j > 0 && ((row_curr >> (j-1)) & 1)) {
                next_row_ids[j] = next_row_ids[j-1];
            } else {
                next_row_ids[j] = (int8_t)(start_node + added_nodes++);
                delta_V++;
            }
            temp &= (temp - 1);
        }

        dsu.init(start_node + added_nodes);

        uint64_t common = row_curr & row_prev;
        while(common) {
            int j = __builtin_ctzll(common);
            if (dsu.unite(next_row_ids[j], c_ids[j])) {
                delta_V--;
            }
            common &= (common - 1);
        }

        current_V += delta_V;

        // Canonicalize IDs
        int max_used = start_node + added_nodes;
        memset(mapping, -1, max_used);
        int remapped_count = 0;

        // Remap boundary IDs
        temp = row_mid;
        while(temp) {
            int j = __builtin_ctzll(temp);
            int root = dsu.find(b_ids[j]);
            if (mapping[root] == -1) mapping[root] = (int8_t)remapped_count++;
            b_ids[j] = mapping[root];
            temp &= (temp - 1);
        }

        // Remap current row IDs
        temp = row_curr;
        while(temp) {
            int j = __builtin_ctzll(temp);
            int root = dsu.find(next_row_ids[j]);
            if (mapping[root] == -1) mapping[root] = (int8_t)remapped_count++;
            c_ids[j] = mapping[root];
            temp &= (temp - 1);
        }

        next_id = remapped_count;

        Run& last = runs[n_runs - 1];
        if (memcmp(last.partition, b_ids, M) == 0) {
            last.count++;
            last.sum_comps += current_V;
        } else {
            Run& next_run = runs[n_runs++];
            memcpy(next_run.partition, b_ids, M);
            // Calculate max ID for next run
            int max_p = -1;
            uint64_t t = row_mid;
            while(t) {
                int j = __builtin_ctzll(t);
                if (b_ids[j] > max_p) max_p = b_ids[j];
                t &= (t - 1);
            }
            next_run.num_ids = (int8_t)(max_p + 1);
            next_run.count = 1;
            next_run.sum_comps = current_V;
        }
    }
    return n_runs;
}

// Generate runs for the right side (expanding from mid+1 up to R)
int generate_right_runs(int mid, int R, Run* runs) {
    int8_t b_ids[MAX_M];
    int8_t c_ids[MAX_M];
    int8_t next_row_ids[MAX_M];
    int8_t mapping[128];
    TinyDSU dsu;

    uint64_t row_midp1 = bit_grid[mid];
    int next_id = 0;
    int current_V = 0;

    memset(b_ids, -1, M);
    uint64_t temp = row_midp1;
    while(temp) {
        int j = __builtin_ctzll(temp);
        if (j > 0 && ((row_midp1 >> (j-1)) & 1)) {
            b_ids[j] = b_ids[j-1];
        } else {
            b_ids[j] = (int8_t)next_id++;
            current_V++;
        }
        temp &= (temp - 1);
    }
    memcpy(c_ids, b_ids, M);

    int n_runs = 0;
    memcpy(runs[0].partition, b_ids, M);
    runs[0].num_ids = (int8_t)next_id;
    runs[0].count = 1;
    runs[0].sum_comps = current_V;
    n_runs = 1;

    for (int i = mid + 2; i <= R; ++i) {
        uint64_t row_curr = bit_grid[i-1];
        uint64_t row_prev = bit_grid[i-2]; // row i-1

        int start_node = next_id;
        int added_nodes = 0;
        int delta_V = 0;

        temp = row_curr;
        while(temp) {
            int j = __builtin_ctzll(temp);
            if (j > 0 && ((row_curr >> (j-1)) & 1)) {
                next_row_ids[j] = next_row_ids[j-1];
            } else {
                next_row_ids[j] = (int8_t)(start_node + added_nodes++);
                delta_V++;
            }
            temp &= (temp - 1);
        }

        dsu.init(start_node + added_nodes);

        uint64_t common = row_curr & row_prev;
        while(common) {
            int j = __builtin_ctzll(common);
            if (dsu.unite(next_row_ids[j], c_ids[j])) {
                delta_V--;
            }
            common &= (common - 1);
        }

        current_V += delta_V;

        int max_used = start_node + added_nodes;
        memset(mapping, -1, max_used);
        int remapped_count = 0;

        temp = row_midp1;
        while(temp) {
            int j = __builtin_ctzll(temp);
            int root = dsu.find(b_ids[j]);
            if (mapping[root] == -1) mapping[root] = (int8_t)remapped_count++;
            b_ids[j] = mapping[root];
            temp &= (temp - 1);
        }

        temp = row_curr;
        while(temp) {
            int j = __builtin_ctzll(temp);
            int root = dsu.find(next_row_ids[j]);
            if (mapping[root] == -1) mapping[root] = (int8_t)remapped_count++;
            c_ids[j] = mapping[root];
            temp &= (temp - 1);
        }

        next_id = remapped_count;

        Run& last = runs[n_runs - 1];
        if (memcmp(last.partition, b_ids, M) == 0) {
            last.count++;
            last.sum_comps += current_V;
        } else {
            Run& next_run = runs[n_runs++];
            memcpy(next_run.partition, b_ids, M);
            int max_p = -1;
            uint64_t t = row_midp1;
            while(t) {
                int j = __builtin_ctzll(t);
                if (b_ids[j] > max_p) max_p = b_ids[j];
                t &= (t - 1);
            }
            next_run.num_ids = (int8_t)(max_p + 1);
            next_run.count = 1;
            next_run.sum_comps = current_V;
        }
    }
    return n_runs;
}

// Compress runs based on active columns (columns where both sides have grass)
int compress_runs(const Run* runs, int n_runs, const int* active_cols, int n_active, CompactRun* out_compact) {
    if (n_runs == 0) return 0;
    
    int8_t map_arr[128];
    
    for(int i=0; i<n_runs; ++i) {
        const Run& r = runs[i];
        CompactRun& cr = out_compact[i];
        cr.total_count = r.count;
        
        // Build signature by remapping IDs on active columns
        memset(map_arr, -1, r.num_ids); 
        int next_id = 0;
        
        for(int k=0; k<n_active; ++k) {
            int col = active_cols[k];
            int val = r.partition[col];
            if (map_arr[val] == -1) {
                map_arr[val] = (int8_t)next_id++;
            }
            cr.sig[k] = map_arr[val];
        }
        cr.num_unique = (int8_t)next_id;
        
        // Zero out remaining part of sig for consistent comparison
        if (n_active < M) {
             memset(cr.sig + n_active, 0, M - n_active);
        }
    }
    
    // Sort to group identical signatures
    std::sort(out_compact, out_compact + n_runs);
    
    // Aggregate counts for identical signatures
    int out_idx = 0;
    for(int i=1; i<n_runs; ++i) {
        if (memcmp(out_compact[out_idx].sig, out_compact[i].sig, M) == 0) {
            out_compact[out_idx].total_count += out_compact[i].total_count;
        } else {
            out_idx++;
            out_compact[out_idx] = out_compact[i];
        }
    }
    return out_idx + 1;
}

long long solve(int L, int R) {
    if (L > R) return 0;
    if (L == R) return count_segments(bit_grid[L-1]);

    int mid = L + (R - L) / 2;
    long long ans = 0;
    long long ans_left = 0, ans_right = 0;

    // Parallel recursion for large enough ranges
    if (R - L > 256) {
        parlay::par_do(
            [&] { ans_left = solve(L, mid); },
            [&] { ans_right = solve(mid + 1, R); }
        );
    } else {
        ans_left = solve(L, mid);
        ans_right = solve(mid + 1, R);
    }
    ans = ans_left + ans_right;

    Run left_runs[MAX_RUNS];
    Run right_runs[MAX_RUNS];
    int n_left = 0, n_right = 0;

    // Parallel run generation
    if (R - L > 256) {
        parlay::par_do(
            [&] { n_left = generate_left_runs(L, mid, left_runs); },
            [&] { n_right = generate_right_runs(mid, R, right_runs); }
        );
    } else {
        n_left = generate_left_runs(L, mid, left_runs);
        n_right = generate_right_runs(mid, R, right_runs);
    }

    uint64_t connect_mask = bit_grid[mid-1] & bit_grid[mid];
    
    // Calculate base contribution
    long long S_L = 0, T_L = 0;
    for(int i=0; i<n_left; ++i) {
        T_L += left_runs[i].count;
        S_L += left_runs[i].sum_comps;
    }
    long long S_R = 0, T_R = 0;
    for(int i=0; i<n_right; ++i) {
        T_R += right_runs[i].count;
        S_R += right_runs[i].sum_comps;
    }

    ans += S_L * T_R + S_R * T_L;

    // Subtract merges
    if (connect_mask != 0) {
        int active_cols[MAX_M];
        int n_active = 0;
        uint64_t m = connect_mask;
        while(m) {
            int j = __builtin_ctzll(m);
            active_cols[n_active++] = j;
            m &= (m - 1);
        }

        // Compress runs to speed up merging
        CompactRun c_left[MAX_RUNS];
        CompactRun c_right[MAX_RUNS];
        int n_c_left = compress_runs(left_runs, n_left, active_cols, n_active, c_left);
        int n_c_right = compress_runs(right_runs, n_right, active_cols, n_active, c_right);

        TinyDSU merge_dsu;
        long long total_merges = 0;

        for(int i=0; i<n_c_left; ++i) {
            const auto& l = c_left[i];
            for(int j=0; j<n_c_right; ++j) {
                const auto& r = c_right[j];
                
                long long pairs = l.total_count * r.total_count;
                int kL = l.num_unique;
                int kR = r.num_unique;
                
                merge_dsu.init(kL + kR);
                int merges = 0;
                
                for(int k=0; k<n_active; ++k) {
                    // Unite active components. Right IDs offset by kL.
                    if (merge_dsu.unite(l.sig[k], kL + r.sig[k])) {
                        merges++;
                    }
                }
                
                total_merges += merges * pairs;
            }
        }
        ans -= total_merges;
    }

    return ans;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (cin >> N >> M) {
        bit_grid.resize(N);
        string row_str;
        row_str.reserve(M);
        for(int i=0; i<N; ++i) {
            cin >> row_str;
            uint64_t row_val = 0;
            for(int j=0; j<M; ++j) {
                if (row_str[j] == '1') {
                    row_val |= (1ULL << j);
                }
            }
            bit_grid[i] = row_val;
        }

        cout << solve(1, N) << endl;
    }
    return 0;
}