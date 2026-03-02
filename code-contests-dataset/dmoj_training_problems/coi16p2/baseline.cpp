//EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <map>

#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Disjoint Set Union (DSU) structure
struct DSU {
    vector<int> parent;
    DSU(int n = 0) {
        if (n > 0) init(n);
    }
    void init(int n) {
        parent.resize(n);
        iota(parent.begin(), parent.end(), 0);
    }
    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]);
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

// Canonicalize a partition vector: renumber component IDs to 0, 1, 2... based on first appearance
// -1 indicates water (no component).
vector<int> canonicalize(const vector<int>& p) {
    int m = p.size();
    int max_id = -1;
    for(int x : p) if(x > max_id) max_id = x;
    
    vector<int> mapping;
    if (max_id >= 0) mapping.assign(max_id + 1, -1);
    
    vector<int> res(m);
    int next_id = 0;
    for (int i = 0; i < m; ++i) {
        if (p[i] == -1) {
            res[i] = -1;
        } else {
            if (mapping[p[i]] == -1) {
                mapping[p[i]] = next_id++;
            }
            res[i] = mapping[p[i]];
        }
    }
    return res;
}

struct Run {
    vector<int> partition;
    long long count;
};

int n, m;
vector<string> grid;

long long solve(int L, int R) {
    if (L > R) return 0;
    if (L == R) {
        int comps = 0;
        for (int j = 0; j < m; ++j) {
            if (grid[L-1][j] == '1') {
                if (j == 0 || grid[L-1][j-1] == '0') comps++;
            }
        }
        return comps;
    }

    int mid = L + (R - L) / 2;
    
    long long ans = 0;
    long long ans_left = 0, ans_right = 0;
    
    // Parallel recursive calls
    parlay::par_do(
        [&] { ans_left = solve(L, mid); },
        [&] { ans_right = solve(mid + 1, R); }
    );
    ans = ans_left + ans_right;

    // Scan Left: i from mid down to L
    vector<Run> left_runs;
    long long sum_C_left = 0;
    
    // Block for Left Scan
    {
        vector<int> row_mid_ids(m, -1);
        int current_comps = 0;
        int next_id = 0;
        
        // Initialize with row mid
        for (int j = 0; j < m; ++j) {
            if (grid[mid-1][j] == '1') {
                if (j > 0 && grid[mid-1][j-1] == '1') {
                    row_mid_ids[j] = row_mid_ids[j-1];
                } else {
                    row_mid_ids[j] = next_id++;
                    current_comps++;
                }
            }
        }
        
        sum_C_left += current_comps;
        left_runs.push_back({canonicalize(row_mid_ids), 1});
        
        vector<int> row_top_ids = row_mid_ids; // Tracks IDs of the current top row (initially mid)
        
        for (int i = mid - 1; i >= L; --i) {
            int max_id = -1;
            for(int x : row_top_ids) if(x > max_id) max_id = x;
            for(int x : row_mid_ids) if(x > max_id) max_id = x;
            int start_new_ids = max_id + 1;
            
            DSU step_dsu(start_new_ids + m);
            
            vector<int> row_i_ids(m, -1);
            int added_grass = 0;
            
            for(int j=0; j<m; ++j) {
                if(grid[i-1][j] == '1') {
                    row_i_ids[j] = start_new_ids + j;
                    added_grass++;
                }
            }
            
            // Horizontal merges in row i
            for(int j=1; j<m; ++j) {
                if(grid[i-1][j] == '1' && grid[i-1][j-1] == '1') {
                    if(step_dsu.unite(row_i_ids[j-1], row_i_ids[j])) {
                        added_grass--;
                    }
                }
            }
            
            // Vertical merges between row i and row i+1
            for(int j=0; j<m; ++j) {
                if(grid[i-1][j] == '1' && grid[i][j] == '1') {
                    if (row_top_ids[j] != -1) {
                         if(step_dsu.unite(row_i_ids[j], row_top_ids[j])) {
                             added_grass--;
                         }
                    }
                }
            }
            
            current_comps += added_grass;
            sum_C_left += current_comps;
            
            // Compress state
            vector<int> mapping(start_new_ids + m, -1);
            int new_next_id = 0;
            
            auto get_mapped = [&](int old_root) {
                if (mapping[old_root] == -1) mapping[old_root] = new_next_id++;
                return mapping[old_root];
            };
            
            for(int j=0; j<m; ++j) {
                if(row_i_ids[j] != -1) row_i_ids[j] = get_mapped(step_dsu.find(row_i_ids[j]));
            }
            for(int j=0; j<m; ++j) {
                if(row_mid_ids[j] != -1) row_mid_ids[j] = get_mapped(step_dsu.find(row_mid_ids[j]));
            }
            
            row_top_ids = row_i_ids;
            
            vector<int> canon_p = canonicalize(row_mid_ids);
            if (left_runs.back().partition == canon_p) {
                left_runs.back().count++;
            } else {
                left_runs.push_back({canon_p, 1});
            }
        }
    }

    // Scan Right: j from mid+1 to R
    vector<Run> right_runs;
    long long sum_C_right = 0;
    
    // Block for Right Scan
    {
        if (mid + 1 <= R) {
            vector<int> row_midp1_ids(m, -1);
            int current_comps = 0;
            int next_id = 0;
            
            for (int j = 0; j < m; ++j) {
                if (grid[mid][j] == '1') {
                    if (j > 0 && grid[mid][j-1] == '1') {
                        row_midp1_ids[j] = row_midp1_ids[j-1];
                    } else {
                        row_midp1_ids[j] = next_id++;
                        current_comps++;
                    }
                }
            }
            
            sum_C_right += current_comps;
            right_runs.push_back({canonicalize(row_midp1_ids), 1});
            
            vector<int> row_bot_ids = row_midp1_ids;
            vector<int> row_fixed_ids = row_midp1_ids; // Boundary row (mid+1)
            
            for (int i = mid + 2; i <= R; ++i) {
                int max_id = -1;
                for(int x : row_bot_ids) if(x > max_id) max_id = x;
                for(int x : row_fixed_ids) if(x > max_id) max_id = x;
                int start_new_ids = max_id + 1;
                
                DSU step_dsu(start_new_ids + m);
                
                vector<int> row_i_ids(m, -1);
                int added_grass = 0;
                
                for(int j=0; j<m; ++j) {
                    if(grid[i-1][j] == '1') {
                        row_i_ids[j] = start_new_ids + j;
                        added_grass++;
                    }
                }
                
                for(int j=1; j<m; ++j) {
                    if(grid[i-1][j] == '1' && grid[i-1][j-1] == '1') {
                        if(step_dsu.unite(row_i_ids[j-1], row_i_ids[j])) added_grass--;
                    }
                }
                
                for(int j=0; j<m; ++j) {
                    if(grid[i-1][j] == '1' && grid[i-2][j] == '1') {
                        if(row_bot_ids[j] != -1) {
                            if(step_dsu.unite(row_i_ids[j], row_bot_ids[j])) added_grass--;
                        }
                    }
                }
                
                current_comps += added_grass;
                sum_C_right += current_comps;
                
                vector<int> mapping(start_new_ids + m, -1);
                int new_next_id = 0;
                auto get_mapped = [&](int old_root) {
                    if (mapping[old_root] == -1) mapping[old_root] = new_next_id++;
                    return mapping[old_root];
                };
                
                for(int j=0; j<m; ++j) {
                    if(row_i_ids[j] != -1) row_i_ids[j] = get_mapped(step_dsu.find(row_i_ids[j]));
                }
                for(int j=0; j<m; ++j) {
                    if(row_fixed_ids[j] != -1) row_fixed_ids[j] = get_mapped(step_dsu.find(row_fixed_ids[j]));
                }
                
                row_bot_ids = row_i_ids;
                
                vector<int> canon_p = canonicalize(row_fixed_ids);
                if (right_runs.back().partition == canon_p) {
                    right_runs.back().count++;
                } else {
                    right_runs.push_back({canon_p, 1});
                }
            }
        }
    }
    
    ans += sum_C_left * (R - mid);
    ans += sum_C_right * (mid - L + 1);
    
    for (const auto& rL : left_runs) {
        for (const auto& rR : right_runs) {
            long long pairs = rL.count * rR.count;
            if (pairs == 0) continue;
            
            const vector<int>& pL = rL.partition;
            const vector<int>& pR = rR.partition;
            
            int max_L = -1; 
            for(int x : pL) if(x > max_L) max_L = x;
            int num_L = max_L + 1;
            
            int max_R = -1;
            for(int x : pR) if(x > max_R) max_R = x;
            int num_R = max_R + 1;
            
            DSU merge_dsu(num_L + num_R);
            int merges = 0;
            
            for(int j=0; j<m; ++j) {
                if (grid[mid-1][j] == '1' && grid[mid][j] == '1') {
                    int u = pL[j];
                    int v = pR[j];
                    if (u != -1 && v != -1) {
                        if (merge_dsu.unite(u, num_L + v)) {
                            merges++;
                        }
                    }
                }
            }
            
            ans -= merges * pairs;
        }
    }
    
    return ans;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    if (!(cin >> n >> m)) return 0;
    grid.resize(n);
    for(int i=0; i<n; ++i) cin >> grid[i];
    
    cout << solve(1, n) << endl;
    
    return 0;
}
//EVOLVE-BLOCK-END