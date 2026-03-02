/**
 * Robots - Competitive Programming Solution
 * 
 * Algorithm:
 * 1. Functional Graph Modeling:
 *    - Nodes: (Field ID, Incoming Direction).
 *    - Edges: Deterministic movement to the next field.
 * 2. Optimized Lookups:
 *    - Use flattened arrays (CSR-like) for row/column field lookups.
 *    - Access coordinates directly from CSR arrays to avoid cache misses on 'fields'.
 * 3. Graph Analysis:
 *    - Decomposition into components (Cycle + Trees).
 *    - Precompute depths, roots, and DFS entry/exit times for O(1) ancestry/distance checks.
 * 4. Parallel Query Processing:
 *    - Use stack-allocated arrays to avoid heap overhead.
 *    - Check 0-turn reachability.
 *    - Calculate min turns using graph properties.
 * 
 * Complexity: O((K + Q) log K)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>

using namespace std;

// Fast IO
namespace FastIO {
    char buffer[1 << 20];
    int buf_pos = 0, buf_len = 0;
    
    inline char read_char() {
        if (buf_pos >= buf_len) {
            buf_pos = 0;
            buf_len = fread(buffer, 1, sizeof(buffer), stdin);
            if (buf_len == 0) return 0;
        }
        return buffer[buf_pos++];
    }
    
    template<typename T>
    inline bool read_int(T &x) {
        char c = read_char();
        while (c <= 32) {
            if (c == 0) return false;
            c = read_char();
        }
        x = 0;
        while (c > 32) {
            x = x * 10 + (c - '0');
            c = read_char();
        }
        return true;
    }
    
    inline bool read_char_val(char &x) {
        char c = read_char();
        while (c <= 32) {
            if (c == 0) return false;
            c = read_char();
        }
        x = c;
        return true;
    }

    char out_buffer[1 << 20];
    int out_pos = 0;

    inline void flush_out() {
        fwrite(out_buffer, 1, out_pos, stdout);
        out_pos = 0;
    }

    inline void write_char(char c) {
        if (out_pos >= sizeof(out_buffer)) flush_out();
        out_buffer[out_pos++] = c;
    }

    template<typename T>
    inline void write_int(T x) {
        if (x < 0) {
            write_char('-');
            x = -x;
        }
        if (x == 0) {
            write_char('0');
            return;
        }
        char temp[20];
        int len = 0;
        while (x > 0) {
            temp[len++] = (x % 10) + '0';
            x /= 10;
        }
        while (len > 0) write_char(temp[--len]);
    }
}

using namespace FastIO;

struct Field {
    int r, c;
    char type;
};

int N, M, K, Q;
vector<Field> fields;

// CSR Lookups
// Row: stores (col, id) sorted by col
vector<int> row_starts;
vector<int> row_coords; // Stores column indices
vector<int> row_ids;

// Col: stores (row, id) sorted by row
vector<int> col_starts;
vector<int> col_coords; // Stores row indices
vector<int> col_ids;

// Graph
vector<int> adj;
vector<int> rev_adj_start;
vector<int> rev_adj_edges;

// Components
vector<int> cycle_id;      // -1 if not on cycle
vector<int> pos_in_cycle;
vector<int> cycle_len;
vector<int> cycle_root;    // For tree nodes: which cycle node (or sink) is the root
vector<int> dist_to_root;
vector<int> tin, tout;
int timer_dfs;
vector<bool> on_cycle;

inline pair<int, int> get_next_field(int r, int c, int dir) {
    if (dir == 0) { // Up
        int start = col_starts[c];
        int end = col_starts[c+1];
        if (start == end) return {-1, -1};
        
        const int* ptr = col_coords.data();
        // first element >= r
        auto it = std::lower_bound(ptr + start, ptr + end, r);
        int idx = (int)(it - ptr);
        
        if (idx == start) { // wrap to bottom
            int fid = col_ids[end - 1];
            return {fid, r + (N - col_coords[end - 1])};
        } else {
            int fid = col_ids[idx - 1];
            return {fid, r - col_coords[idx - 1]};
        }
    } else if (dir == 2) { // Down
        int start = col_starts[c];
        int end = col_starts[c+1];
        if (start == end) return {-1, -1};
        
        const int* ptr = col_coords.data();
        // first element > r
        auto it = std::upper_bound(ptr + start, ptr + end, r);
        int idx = (int)(it - ptr);
        
        if (idx == end) { // wrap to top
            int fid = col_ids[start];
            return {fid, (N - r) + col_coords[start]};
        } else {
            int fid = col_ids[idx];
            return {fid, col_coords[idx] - r};
        }
    } else if (dir == 1) { // Right
        int start = row_starts[r];
        int end = row_starts[r+1];
        if (start == end) return {-1, -1};
        
        const int* ptr = row_coords.data();
        // first element > c
        auto it = std::upper_bound(ptr + start, ptr + end, c);
        int idx = (int)(it - ptr);
        
        if (idx == end) { // wrap to left
            int fid = row_ids[start];
            return {fid, (M - c) + row_coords[start]};
        } else {
            int fid = row_ids[idx];
            return {fid, row_coords[idx] - c};
        }
    } else { // Left
        int start = row_starts[r];
        int end = row_starts[r+1];
        if (start == end) return {-1, -1};
        
        const int* ptr = row_coords.data();
        // first element >= c
        auto it = std::lower_bound(ptr + start, ptr + end, c);
        int idx = (int)(it - ptr);
        
        if (idx == start) { // wrap to right
            int fid = row_ids[end - 1];
            return {fid, c + (M - row_coords[end - 1])};
        } else {
            int fid = row_ids[idx - 1];
            return {fid, c - row_coords[idx - 1]};
        }
    }
}

inline int get_out_dir(int in_dir, char type) {
    return (type == 'L') ? (in_dir + 3) & 3 : (in_dir + 1) & 3;
}

inline int get_in_dir(int out_dir, char type) {
    return (type == 'L') ? (out_dir + 1) & 3 : (out_dir + 3) & 3;
}

inline bool is_on_segment(int r, int c, int tr, int tc, int dir, int dist_to_next) {
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

// Iterative DFS
void process_tree(int start_node) {
    static vector<pair<int, int>> stack;
    if (stack.capacity() < 2048) stack.reserve(2048);
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
    if (!read_int(N)) return 0;
    read_int(M);
    read_int(K);
    
    fields.resize(K);
    parlay::sequence<int> indices(K);
    for (int i = 0; i < K; ++i) {
        read_int(fields[i].r);
        read_int(fields[i].c);
        read_char_val(fields[i].type);
        indices[i] = i;
    }
    
    // Sort by Row then Col
    parlay::sort_inplace(indices, [&](int a, int b) {
        if (fields[a].r != fields[b].r) return fields[a].r < fields[b].r;
        return fields[a].c < fields[b].c;
    });
    
    row_starts.assign(N + 2, 0);
    row_coords.resize(K);
    row_ids.resize(K);
    
    // Fill Row CSR
    {
        int current_r = 0;
        for (int i = 0; i < K; ++i) {
            int idx = indices[i];
            int r = fields[idx].r;
            while (current_r < r) {
                row_starts[++current_r] = i;
            }
            row_coords[i] = fields[idx].c;
            row_ids[i] = idx;
        }
        while (current_r <= N) {
            row_starts[++current_r] = K;
        }
    }
    
    // Sort by Col then Row
    parlay::sort_inplace(indices, [&](int a, int b) {
        if (fields[a].c != fields[b].c) return fields[a].c < fields[b].c;
        return fields[a].r < fields[b].r;
    });
    
    col_starts.assign(M + 2, 0);
    col_coords.resize(K);
    col_ids.resize(K);
    
    // Fill Col CSR
    {
        int current_c = 0;
        for (int i = 0; i < K; ++i) {
            int idx = indices[i];
            int c = fields[idx].c;
            while (current_c < c) {
                col_starts[++current_c] = i;
            }
            col_coords[i] = fields[idx].r;
            col_ids[i] = idx;
        }
        while (current_c <= M) {
            col_starts[++current_c] = K;
        }
    }
    
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
    
    // Build Reverse Graph
    vector<int> in_degree(num_nodes, 0);
    for (int u = 0; u < num_nodes; ++u) {
        if (adj[u] != -1) in_degree[adj[u]]++;
    }
    
    rev_adj_start.assign(num_nodes + 1, 0);
    int current = 0;
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
    
    // Component Analysis
    cycle_id.assign(num_nodes, -1);
    pos_in_cycle.assign(num_nodes, -1);
    cycle_root.assign(num_nodes, -1);
    dist_to_root.assign(num_nodes, 0);
    tin.assign(num_nodes, 0);
    tout.assign(num_nodes, 0);
    on_cycle.assign(num_nodes, false);
    
    // Use degree-based topological sort to find cycles
    // Reuse in_degree for topological sort
    fill(in_degree.begin(), in_degree.end(), 0);
    for (int u = 0; u < num_nodes; ++u) {
        if (adj[u] != -1) in_degree[adj[u]]++;
    }
    
    vector<int> q; q.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        if (in_degree[i] == 0) q.push_back(i);
    }
    
    int head = 0;
    while(head < q.size()) {
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
            int len = 0;
            // First pass: mark and count
            while (cycle_id[curr] == -1) {
                cycle_id[curr] = cycle_cnt;
                curr = adj[curr];
                len++;
            }
            cycle_len.push_back(len);
            
            // Second pass: set properties
            curr = i;
            for (int k = 0; k < len; ++k) {
                pos_in_cycle[curr] = k;
                on_cycle[curr] = true;
                cycle_root[curr] = curr;
                dist_to_root[curr] = 0;
                curr = adj[curr];
            }
            cycle_cnt++;
        }
    }
    
    timer_dfs = 0;
    
    for (int i = 0; i < num_nodes; ++i) {
        if (on_cycle[i]) process_tree(i);
    }
    for (int i = 0; i < num_nodes; ++i) {
        if (adj[i] == -1) process_tree(i);
    }
    
    read_int(Q);
    struct Query {
        int r1, c1, r2, c2, id;
    };
    vector<Query> queries(Q);
    for(int i=0; i<Q; ++i) {
        read_int(queries[i].r1);
        read_int(queries[i].c1);
        read_int(queries[i].r2);
        read_int(queries[i].c2);
        queries[i].id = i;
    }
    
    vector<int> results(Q);
    
    parlay::parallel_for(0, Q, [&](int i) {
        const auto& q = queries[i];
        if (q.r1 == q.r2 && q.c1 == q.c2) {
            results[q.id] = 0;
            return;
        }
        
        int ans = 2e9;
        int entries[4], num_entries = 0;
        
        for (int d = 0; d < 4; ++d) {
            pair<int, int> next = get_next_field(q.r1, q.c1, d);
            if (is_on_segment(q.r1, q.c1, q.r2, q.c2, d, next.second)) {
                ans = 0;
            }
            if (next.first != -1) {
                entries[num_entries++] = 4 * next.first + d;
            }
        }
        
        if (ans == 0) {
            results[q.id] = 0;
            return;
        }
        
        int feeders[4], num_feeders = 0;
        for (int d_arr = 0; d_arr < 4; ++d_arr) {
            int d_look = (d_arr + 2) & 3;
            pair<int, int> prev = get_next_field(q.r2, q.c2, d_look);
            if (prev.first != -1) {
                int d_in = get_in_dir(d_arr, fields[prev.first].type);
                feeders[num_feeders++] = 4 * prev.first + d_in;
            }
        }
        
        for (int k = 0; k < num_entries; ++k) {
            int u = entries[k];
            for (int m = 0; m < num_feeders; ++m) {
                int v = feeders[m];
                
                int dist = -1;
                
                if (on_cycle[v]) {
                    if (cycle_root[u] != -1 && cycle_id[cycle_root[u]] == cycle_id[v]) {
                        int root = cycle_root[u];
                        int clen = cycle_len[cycle_id[v]];
                        int diff = (pos_in_cycle[v] - pos_in_cycle[root] + clen);
                        if (diff >= clen) diff -= clen;
                        dist = dist_to_root[u] + diff;
                    }
                } else {
                    if (cycle_root[u] != -1 && cycle_root[u] == cycle_root[v]) {
                        if (tin[v] <= tin[u] && tout[u] <= tout[v]) {
                            dist = dist_to_root[u] - dist_to_root[v];
                        }
                    }
                }
                
                if (dist != -1) {
                    int cand = 1 + dist;
                    if (cand < ans) ans = cand;
                }
            }
        }
        
        results[q.id] = (ans == 2e9) ? -1 : ans;
    });
    
    for(int i=0; i<Q; ++i) {
        write_int(results[i]);
        write_char('\n');
    }
    flush_out();
    
    return 0;
}