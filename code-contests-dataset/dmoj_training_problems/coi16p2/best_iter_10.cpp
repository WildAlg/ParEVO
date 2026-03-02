/**
 * Solution for Meadow Complexity
 * 
 * Algorithm: Divide and Conquer on rows with Parallelism.
 * 
 * We recursively divide the row range [L, R] into [L, mid] and [mid+1, R].
 * The total sum consists of:
 * 1. Sum of complexities entirely within [L, mid].
 * 2. Sum of complexities entirely within [mid+1, R].
 * 3. Sum of complexities for ranges crossing the boundary.
 * 
 * For the crossing ranges, we perform a linear sweep from the boundary outwards to generate "Runs".
 * Each Run represents a sequence of rows where the connectivity partition of the boundary row remains identical.
 * We efficiently combine Runs from the left and right sides to compute the total complexity.
 * 
 * Optimizations:
 * - Flattened grid representation (1D vector) for better cache locality.
 * - Fixed-size arrays and POD structs for Runs to avoid heap allocations in inner loops.
 * - Tiny, fixed-size DSU and Mapper structures reused on the stack.
 * - Parallel execution using parlay::par_do.
 * 
 * Time Complexity: O(N * M * log N)
 * Space Complexity: O(N * M)
 */

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

using namespace std;

// Global constants and data
const int MAX_M = 50;
int N, M;
vector<char> flat_grid;

// Run structure with fixed size array to avoid allocations
struct Run {
    int partition[MAX_M];
    int num_comps;
    long long count;
    long long sum_V;
};

// Small DSU with fixed size
struct TinyDSU {
    int parent[2 * MAX_M + 10]; // 2*50 + margin
    int n;
    
    void init(int _n) {
        n = _n;
        for(int i=0; i<n; ++i) parent[i] = i;
    }
    
    int find(int i) {
        int root = i;
        while(root != parent[root]) root = parent[root];
        int curr = i;
        while(curr != root) {
            int nxt = parent[curr];
            parent[curr] = root;
            curr = nxt;
        }
        return root;
    }
    
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = root_j;
            return true;
        }
        return false;
    }
};

// Mapper for ID canonicalization
struct Mapper {
    int map_arr[2 * MAX_M + 10];
    
    Mapper() {
        fill(begin(map_arr), end(map_arr), -1);
    }
    
    // Reset only up to max_idx used to save time
    void reset(int max_idx) {
        for(int i=0; i <= max_idx; ++i) map_arr[i] = -1;
    }
};

long long solve(int L, int R) {
    if (L > R) return 0;
    
    // Base case: Single row
    if (L == R) {
        int comps = 0;
        int r_offset = (L - 1) * M;
        for(int j=0; j<M; ++j) {
            if (flat_grid[r_offset + j] == '1') {
                if (j == 0 || flat_grid[r_offset + j - 1] == '0') comps++;
            }
        }
        return comps;
    }

    int mid = L + (R - L) / 2;
    long long ans = 0;
    
    long long ans_left = 0, ans_right = 0;
    
    // Parallel recursive calls with cutoff for small tasks
    if (R - L < 128) {
        ans_left = solve(L, mid);
        ans_right = solve(mid + 1, R);
    } else {
        parlay::par_do(
            [&] { ans_left = solve(L, mid); },
            [&] { ans_right = solve(mid + 1, R); }
        );
    }
    ans = ans_left + ans_right;

    // --- Left Sweep (mid down to L) ---
    vector<Run> left_runs;
    left_runs.reserve(mid - L + 1);
    
    {
        int b_ids[MAX_M];
        int c_ids[MAX_M];
        int new_row_ids[MAX_M];
        TinyDSU dsu;
        Mapper mapper;
        
        int current_V = 0;
        int next_id = 0;
        int r_mid = mid - 1;
        int offset_mid = r_mid * M;

        // Initialize at row mid
        for(int j=0; j<M; ++j) {
            if (flat_grid[offset_mid + j] == '1') {
                if (j == 0 || flat_grid[offset_mid + j - 1] == '0') {
                    b_ids[j] = next_id++;
                    current_V++;
                } else {
                    b_ids[j] = b_ids[j-1];
                }
            } else {
                b_ids[j] = -1;
            }
        }
        memcpy(c_ids, b_ids, M * sizeof(int));
        
        Run first_run;
        memcpy(first_run.partition, b_ids, M * sizeof(int));
        first_run.num_comps = next_id;
        first_run.count = 1;
        first_run.sum_V = current_V;
        left_runs.push_back(first_run);
        
        for(int i = mid - 1; i >= L; --i) {
            int r_idx = i - 1;
            int offset = r_idx * M;
            int offset_next = i * M; 
            
            int start_node = next_id;
            int added_nodes = 0;
            int row_V_delta = 0;
            
            // 1. Assign new IDs for row i and merge horizontally
            for(int j=0; j<M; ++j) {
                if (flat_grid[offset + j] == '1') {
                    if (j == 0 || flat_grid[offset + j - 1] == '0') {
                        new_row_ids[j] = start_node + added_nodes++;
                        row_V_delta++;
                    } else {
                        new_row_ids[j] = new_row_ids[j-1];
                    }
                } else {
                    new_row_ids[j] = -1;
                }
            }
            
            dsu.init(start_node + added_nodes);
            
            // 2. Vertical merges with row i+1
            for(int j=0; j<M; ++j) {
                if (flat_grid[offset + j] == '1' && flat_grid[offset_next + j] == '1') {
                    if (c_ids[j] != -1) {
                         if (dsu.unite(new_row_ids[j], c_ids[j])) {
                             row_V_delta--;
                         }
                    }
                }
            }
            
            current_V += row_V_delta;
            
            // 3. Remap IDs to keep DSU small
            int remapped_count = 0;
            int max_idx_used = start_node + added_nodes;
            mapper.reset(max_idx_used);
            
            for(int j=0; j<M; ++j) {
                if (b_ids[j] != -1) {
                    int r = dsu.find(b_ids[j]);
                    if (mapper.map_arr[r] == -1) mapper.map_arr[r] = remapped_count++;
                    b_ids[j] = mapper.map_arr[r];
                }
            }
            for(int j=0; j<M; ++j) {
                if (new_row_ids[j] != -1) {
                    int r = dsu.find(new_row_ids[j]);
                    if (mapper.map_arr[r] == -1) mapper.map_arr[r] = remapped_count++;
                    new_row_ids[j] = mapper.map_arr[r];
                }
            }
            
            memcpy(c_ids, new_row_ids, M * sizeof(int));
            next_id = remapped_count;
            
            // Check if run can be extended
            if (memcmp(b_ids, left_runs.back().partition, M * sizeof(int)) == 0) {
                left_runs.back().count++;
                left_runs.back().sum_V += current_V;
            } else {
                int max_p = -1;
                for(int k=0; k<M; ++k) if(b_ids[k] > max_p) max_p = b_ids[k];
                Run new_run;
                memcpy(new_run.partition, b_ids, M * sizeof(int));
                new_run.num_comps = max_p + 1;
                new_run.count = 1;
                new_run.sum_V = current_V;
                left_runs.push_back(new_run);
            }
        }
    }

    // --- Right Sweep (mid+1 up to R) ---
    vector<Run> right_runs;
    right_runs.reserve(R - mid);
    
    {
        int b_ids[MAX_M];
        int c_ids[MAX_M];
        int new_row_ids[MAX_M];
        TinyDSU dsu;
        Mapper mapper;
        
        int current_V = 0;
        int next_id = 0;
        int r_midp1 = mid; 
        int offset_midp1 = r_midp1 * M;
        
        // Initialize at row mid+1
        for(int j=0; j<M; ++j) {
            if (flat_grid[offset_midp1 + j] == '1') {
                if (j == 0 || flat_grid[offset_midp1 + j - 1] == '0') {
                    b_ids[j] = next_id++;
                    current_V++;
                } else {
                    b_ids[j] = b_ids[j-1];
                }
            } else {
                b_ids[j] = -1;
            }
        }
        memcpy(c_ids, b_ids, M * sizeof(int));
        
        Run first_run;
        memcpy(first_run.partition, b_ids, M * sizeof(int));
        first_run.num_comps = next_id;
        first_run.count = 1;
        first_run.sum_V = current_V;
        right_runs.push_back(first_run);
        
        for(int i = mid + 2; i <= R; ++i) {
            int r_idx = i - 1;
            int offset = r_idx * M;
            int offset_prev = (i - 2) * M;
            
            int start_node = next_id;
            int added_nodes = 0;
            int row_V_delta = 0;
            
            for(int j=0; j<M; ++j) {
                if (flat_grid[offset + j] == '1') {
                    if (j == 0 || flat_grid[offset + j - 1] == '0') {
                        new_row_ids[j] = start_node + added_nodes++;
                        row_V_delta++;
                    } else {
                        new_row_ids[j] = new_row_ids[j-1];
                    }
                } else {
                    new_row_ids[j] = -1;
                }
            }
            
            dsu.init(start_node + added_nodes);
            
            for(int j=0; j<M; ++j) {
                if (flat_grid[offset + j] == '1' && flat_grid[offset_prev + j] == '1') {
                    if (c_ids[j] != -1) {
                         if (dsu.unite(new_row_ids[j], c_ids[j])) {
                             row_V_delta--;
                         }
                    }
                }
            }
            
            current_V += row_V_delta;
            
            int remapped_count = 0;
            int max_idx_used = start_node + added_nodes;
            mapper.reset(max_idx_used);
            
            for(int j=0; j<M; ++j) {
                if (b_ids[j] != -1) {
                    int r = dsu.find(b_ids[j]);
                    if (mapper.map_arr[r] == -1) mapper.map_arr[r] = remapped_count++;
                    b_ids[j] = mapper.map_arr[r];
                }
            }
            for(int j=0; j<M; ++j) {
                if (new_row_ids[j] != -1) {
                    int r = dsu.find(new_row_ids[j]);
                    if (mapper.map_arr[r] == -1) mapper.map_arr[r] = remapped_count++;
                    new_row_ids[j] = mapper.map_arr[r];
                }
            }
            
            memcpy(c_ids, new_row_ids, M * sizeof(int));
            next_id = remapped_count;
            
            if (memcmp(b_ids, right_runs.back().partition, M * sizeof(int)) == 0) {
                right_runs.back().count++;
                right_runs.back().sum_V += current_V;
            } else {
                int max_p = -1;
                for(int k=0; k<M; ++k) if(b_ids[k] > max_p) max_p = b_ids[k];
                Run new_run;
                memcpy(new_run.partition, b_ids, M * sizeof(int));
                new_run.num_comps = max_p + 1;
                new_run.count = 1;
                new_run.sum_V = current_V;
                right_runs.push_back(new_run);
            }
        }
    }

    // --- Combine ---
    vector<int> connect_indices;
    connect_indices.reserve(M);
    int offset_up = (mid - 1) * M;
    int offset_down = mid * M;
    for(int j=0; j<M; ++j) {
        if (flat_grid[offset_up + j] == '1' && flat_grid[offset_down + j] == '1') {
            connect_indices.push_back(j);
        }
    }

    if (connect_indices.empty()) {
        long long total_count_R = 0;
        long long total_sum_V_R = 0;
        for(const auto& r : right_runs) {
            total_count_R += r.count;
            total_sum_V_R += r.sum_V;
        }
        for(const auto& l : left_runs) {
            ans += l.sum_V * total_count_R + total_sum_V_R * l.count;
        }
    } else {
        TinyDSU merge_dsu;
        for (const auto& runL : left_runs) {
            for (const auto& runR : right_runs) {
                long long pairs = runL.count * runR.count;
                long long term = runL.sum_V * runR.count + runR.sum_V * runL.count;
                
                int kL = runL.num_comps;
                int kR = runR.num_comps;
                
                merge_dsu.init(kL + kR);
                int merges = 0;
                
                for (int j : connect_indices) {
                    int u = runL.partition[j];
                    int v = runR.partition[j];
                    if (merge_dsu.unite(u, kL + v)) {
                        merges++;
                    }
                }
                
                term -= (long long)merges * pairs;
                ans += term;
            }
        }
    }

    return ans;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    if (cin >> N >> M) {
        flat_grid.resize(N * M);
        for(int i=0; i<N; ++i) {
            string row;
            cin >> row;
            for(int j=0; j<M; ++j) {
                flat_grid[i * M + j] = row[j];
            }
        }
        cout << solve(1, N) << endl;
    }
    return 0;
}