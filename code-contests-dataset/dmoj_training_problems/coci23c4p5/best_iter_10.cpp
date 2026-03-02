/**
 * Robots - Competitive Programming Solution
 * 
 * Algorithm:
 * 1. Model the problem as a functional graph where nodes are (Field, Incoming Direction).
 * 2. Use Coordinate Compression / CSR-like structures for efficient O(log K) spatial lookups.
 * 3. Precompute graph components (Cycles + Trees) for O(1) distance queries.
 * 4. Parallelize construction and query processing using Parlay.
 * 
 * Complexity: O((K + Q) log K)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

using namespace std;

// Fast I/O
void fast_io() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
}

struct Field {
    int r, c;
    char type;
    int id;
};

// Global Data
int N, M, K, Q;
vector<Field> fields;

// Lookup structures
// We use Compressed Sparse Row (CSR) style structures for efficient row/column lookups
// fields_by_row[row_start[r] ... row_start[r] + row_count[r] - 1] contains field IDs in row r, sorted by c
vector<int> fields_by_row;
vector<int> row_start;
vector<int> row_count;

// fields_by_col[col_start[c] ... col_start[c] + col_count[c] - 1] contains field IDs in col c, sorted by r
vector<int> fields_by_col;
vector<int> col_start;
vector<int> col_count;

// Graph Data
// Nodes: 0..4K-1. Node u = 4*field_id + incoming_dir
// adj[u] = v (Functional graph: max 1 outgoing edge)
vector<int> adj;

// Reverse graph in CSR format for efficient tree traversal
vector<int> rev_adj_start;
vector<int> rev_adj_edges;

// Component Data
vector<int> comp;          // Component ID
vector<bool> on_cycle;     // Is node on cycle
vector<int> cycle_root;    // Root on cycle (or sink for dead-end trees)
vector<int> dist_to_root;  // Distance to root
vector<int> cycle_id;      // Cycle ID
vector<int> pos_in_cycle;  // Position in cycle
vector<int> cycle_len;     // Length of cycle
vector<int> tin, tout;     // DFS entry/exit times
int timer_dfs;

// Helper: Get next field and distance
// Returns {field_id, dist}
// dist is distance on grid. If no field, returns {-1, -1}
pair<int, int> get_next_field(int r, int c, int dir) {
    if (dir == 0) { // Up: decreasing row, same col
        if (col_count[c] == 0) return {-1, -1};
        int start = col_start[c];
        int cnt = col_count[c];
        
        // Find first field with row >= r
        auto it = std::lower_bound(fields_by_col.begin() + start, fields_by_col.begin() + start + cnt, r, 
            [](int fid, int val) { return fields[fid].r < val; });
            
        int idx_in_vec = distance(fields_by_col.begin(), it);
        
        if (idx_in_vec == start) {
            // Wrap around to the last element in this column
            int fid = fields_by_col[start + cnt - 1];
            int dist = r + (N - fields[fid].r);
            return {fid, dist};
        } else {
            int fid = fields_by_col[idx_in_vec - 1];
            int dist = r - fields[fid].r;
            return {fid, dist};
        }
    } else if (dir == 2) { // Down: increasing row, same col
        if (col_count[c] == 0) return {-1, -1};
        int start = col_start[c];
        int cnt = col_count[c];
        
        // Find first field with row > r
        auto it = std::upper_bound(fields_by_col.begin() + start, fields_by_col.begin() + start + cnt, r,
            [](int val, int fid) { return val < fields[fid].r; });
            
        int idx_in_vec = distance(fields_by_col.begin(), it);
        
        if (idx_in_vec == start + cnt) {
            // Wrap around to first
            int fid = fields_by_col[start];
            int dist = (N - r) + fields[fid].r;
            return {fid, dist};
        } else {
            int fid = fields_by_col[idx_in_vec];
            int dist = fields[fid].r - r;
            return {fid, dist};
        }
    } else if (dir == 1) { // Right: increasing col, same row
        if (row_count[r] == 0) return {-1, -1};
        int start = row_start[r];
        int cnt = row_count[r];
        
        // Find first field with col > c
        auto it = std::upper_bound(fields_by_row.begin() + start, fields_by_row.begin() + start + cnt, c,
            [](int val, int fid) { return val < fields[fid].c; });
            
        int idx_in_vec = distance(fields_by_row.begin(), it);
        
        if (idx_in_vec == start + cnt) {
            // Wrap around to first
            int fid = fields_by_row[start];
            int dist = (M - c) + fields[fid].c;
            return {fid, dist};
        } else {
            int fid = fields_by_row[idx_in_vec];
            int dist = fields[fid].c - c;
            return {fid, dist};
        }
    } else { // Left: decreasing col, same row
        if (row_count[r] == 0) return {-1, -1};
        int start = row_start[r];
        int cnt = row_count[r];
        
        // Find first field with col >= c
        auto it = std::lower_bound(fields_by_row.begin() + start, fields_by_row.begin() + start + cnt, c,
            [](int fid, int val) { return fields[fid].c < val; });
            
        int idx_in_vec = distance(fields_by_row.begin(), it);
        
        if (idx_in_vec == start) {
            // Wrap around to last
            int fid = fields_by_row[start + cnt - 1];
            int dist = c + (M - fields[fid].c);
            return {fid, dist};
        } else {
            int fid = fields_by_row[idx_in_vec - 1];
            int dist = c - fields[fid].c;
            return {fid, dist};
        }
    }
}

// Get outgoing direction
inline int get_out_dir(int in_dir, char type) {
    return (type == 'L') ? (in_dir + 3) & 3 : (in_dir + 1) & 3;
}

// Get incoming direction
inline int get_in_dir(int out_dir, char type) {
    return (type == 'L') ? (out_dir + 1) & 3 : (out_dir + 3) & 3;
}

// Check if point is on segment
// dist_to_next is the distance to the next field. If -1, it means infinite.
bool is_on_segment(int r, int c, int tr, int tc, int dir, int dist_to_next) {
    int dist_target = 0;
    if (dir == 0) { // Up
        if (tc != c) return false;
        dist_target = r - tr;
        if (dist_target < 0) dist_target += N;
    } else if (dir == 2) { // Down
        if (tc != c) return false;
        dist_target = tr - r;
        if (dist_target < 0) dist_target += N;
    } else if (dir == 1) { // Right
        if (tr != r) return false;
        dist_target = tc - c;
        if (dist_target < 0) dist_target += M;
    } else { // Left
        if (tr != r) return false;
        dist_target = c - tc;
        if (dist_target < 0) dist_target += M;
    }
    
    if (dist_to_next == -1) return true;
    return dist_target <= dist_to_next;
}

// Iterative DFS for tree processing
void process_tree_iterative(int start_node) {
    static vector<pair<int, int>> stack;
    if (stack.capacity() < 4096) stack.reserve(4096);
    stack.clear();
    
    stack.push_back({start_node, 0});
    tin[start_node] = ++timer_dfs;
    
    while (!stack.empty()) {
        int u = stack.back().first;
        int idx = stack.back().second;
        
        int start_edge = rev_adj_start[u];
        int end_edge = rev_adj_start[u+1];
        
        if (start_edge + idx < end_edge) {
            int v = rev_adj_edges[start_edge + idx];
            stack.back().second++; 
            
            if (!on_cycle[v]) {
                cycle_root[v] = cycle_root[u];
                dist_to_root[v] = dist_to_root[u] + 1;
                tin[v] = ++timer_dfs;
                stack.push_back({v, 0});
            }
        } else {
            tout[u] = ++timer_dfs;
            stack.pop_back();
        }
    }
}

int main() {
    fast_io();
    if (!(cin >> N >> M >> K)) return 0;
    
    fields.resize(K);
    for (int i = 0; i < K; ++i) {
        cin >> fields[i].r >> fields[i].c >> fields[i].type;
        fields[i].id = i;
    }
    
    // Build row/col lookups
    row_count.assign(N + 1, 0);
    col_count.assign(M + 1, 0);
    
    for (const auto& f : fields) {
        row_count[f.r]++;
        col_count[f.c]++;
    }
    
    row_start.assign(N + 2, 0);
    col_start.assign(M + 2, 0);
    
    int current = 0;
    for(int i = 1; i <= N; ++i) {
        int c = row_count[i];
        row_count[i] = 0; 
        row_start[i] = current;
        current += c;
    }
    row_start[N+1] = current; 
    
    current = 0;
    for(int i = 1; i <= M; ++i) {
        int c = col_count[i];
        col_count[i] = 0; 
        col_start[i] = current;
        current += c;
    }
    col_start[M+1] = current;
    
    fields_by_row.resize(K);
    fields_by_col.resize(K);
    
    for (int i = 0; i < K; ++i) {
        int r = fields[i].r;
        int c = fields[i].c;
        fields_by_row[row_start[r] + row_count[r]++] = i;
        fields_by_col[col_start[c] + col_count[c]++] = i;
    }
    
    parlay::parallel_for(1, N + 1, [&](int r) {
        if (row_count[r] > 1) {
            std::sort(fields_by_row.begin() + row_start[r], fields_by_row.begin() + row_start[r] + row_count[r],
                [](int a, int b) { return fields[a].c < fields[b].c; });
        }
    });
    
    parlay::parallel_for(1, M + 1, [&](int c) {
        if (col_count[c] > 1) {
            std::sort(fields_by_col.begin() + col_start[c], fields_by_col.begin() + col_start[c] + col_count[c],
                [](int a, int b) { return fields[a].r < fields[b].r; });
        }
    });
    
    // Build Graph
    int num_nodes = 4 * K;
    adj.assign(num_nodes, -1);
    
    parlay::parallel_for(0, K, [&](int i) {
        for (int d_in = 0; d_in < 4; ++d_in) {
            int u = 4 * i + d_in;
            int d_out = get_out_dir(d_in, fields[i].type);
            pair<int, int> next = get_next_field(fields[i].r, fields[i].c, d_out);
            if (next.first != -1) {
                adj[u] = 4 * next.first + d_out;
            }
        }
    });
    
    // Build Reverse Graph (CSR)
    vector<int> in_degree(num_nodes, 0);
    for (int u = 0; u < num_nodes; ++u) {
        if (adj[u] != -1) in_degree[adj[u]]++;
    }
    
    rev_adj_start.assign(num_nodes + 1, 0);
    current = 0;
    for (int i = 0; i < num_nodes; ++i) {
        int d = in_degree[i];
        in_degree[i] = 0; 
        rev_adj_start[i] = current;
        current += d;
    }
    rev_adj_start[num_nodes] = current;
    rev_adj_edges.resize(current);
    
    for (int u = 0; u < num_nodes; ++u) {
        if (adj[u] != -1) {
            int v = adj[u];
            rev_adj_edges[rev_adj_start[v] + in_degree[v]++] = u;
        }
    }
    
    // Analyze Components
    comp.assign(num_nodes, -1);
    on_cycle.assign(num_nodes, false);
    cycle_root.assign(num_nodes, -1);
    cycle_id.assign(num_nodes, -1);
    pos_in_cycle.assign(num_nodes, -1);
    dist_to_root.assign(num_nodes, 0);
    tin.assign(num_nodes, 0);
    tout.assign(num_nodes, 0);
    
    fill(in_degree.begin(), in_degree.end(), 0);
    for (int u = 0; u < num_nodes; ++u) {
        if (adj[u] != -1) in_degree[adj[u]]++;
    }
    
    vector<int> q; 
    q.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        if (in_degree[i] == 0) q.push_back(i);
    }
    
    int head = 0;
    while(head < q.size()){
        int u = q[head++];
        if (adj[u] != -1) {
            int v = adj[u];
            in_degree[v]--;
            if (in_degree[v] == 0) q.push_back(v);
        }
    }
    
    int cycle_cnt = 0;
    for (int i = 0; i < num_nodes; ++i) {
        if (in_degree[i] > 0 && cycle_id[i] == -1) {
            int curr = i;
            vector<int> path;
            while (cycle_id[curr] == -1) {
                cycle_id[curr] = cycle_cnt;
                path.push_back(curr);
                curr = adj[curr];
            }
            
            int len = path.size();
            cycle_len.push_back(len);
            for (int k = 0; k < len; ++k) {
                int node = path[k];
                pos_in_cycle[node] = k;
                on_cycle[node] = true;
                cycle_root[node] = node;
                dist_to_root[node] = 0;
            }
            cycle_cnt++;
        }
    }
    
    timer_dfs = 0;
    
    for (int i = 0; i < num_nodes; ++i) {
        if (on_cycle[i]) {
            process_tree_iterative(i);
        }
    }
    
    for (int i = 0; i < num_nodes; ++i) {
        if (adj[i] == -1) {
            process_tree_iterative(i);
        }
    }
    
    cin >> Q;
    vector<tuple<int, int, int, int>> queries(Q);
    for(int i=0; i<Q; ++i) {
        cin >> get<0>(queries[i]) >> get<1>(queries[i]) >> get<2>(queries[i]) >> get<3>(queries[i]);
    }
    
    vector<int> results(Q);
    
    parlay::parallel_for(0, Q, [&](int i) {
        int r1, c1, r2, c2;
        tie(r1, c1, r2, c2) = queries[i];
        
        if (r1 == r2 && c1 == c2) {
            results[i] = 0;
            return;
        }
        
        int ans = 2e9;
        
        vector<int> entries;
        for (int d = 0; d < 4; ++d) {
            pair<int, int> next = get_next_field(r1, c1, d);
            if (is_on_segment(r1, c1, r2, c2, d, next.second)) {
                ans = 0;
            }
            if (next.first != -1) {
                entries.push_back(4 * next.first + d);
            }
        }
        
        if (ans == 0) {
            results[i] = 0;
            return;
        }
        
        vector<int> feeders;
        for (int d_arr = 0; d_arr < 4; ++d_arr) {
            int d_look = (d_arr + 2) & 3;
            pair<int, int> prev = get_next_field(r2, c2, d_look);
            if (prev.first != -1) {
                int d_in = get_in_dir(d_arr, fields[prev.first].type);
                feeders.push_back(4 * prev.first + d_in);
            }
        }
        
        for (int u : entries) {
            for (int v : feeders) {
                int d_graph = -1;
                
                if (on_cycle[v]) {
                    if (cycle_root[u] != -1 && cycle_id[cycle_root[u]] == cycle_id[v]) {
                        int root_u = cycle_root[u];
                        int len = cycle_len[cycle_id[v]]; 
                        int diff = (pos_in_cycle[v] - pos_in_cycle[root_u] + len) % len;
                        d_graph = dist_to_root[u] + diff;
                    }
                } 
                else {
                     if (cycle_root[u] != -1 && cycle_root[u] == cycle_root[v]) {
                         if (tin[v] <= tin[u] && tout[u] <= tout[v]) {
                             d_graph = dist_to_root[u] - dist_to_root[v];
                         }
                     }
                }
                
                if (d_graph != -1) {
                    ans = min(ans, 1 + d_graph);
                }
            }
        }
        
        results[i] = (ans == 2e9) ? -1 : ans;
    });
    
    for(int x : results) cout << x << "\n";
    
    return 0;
}