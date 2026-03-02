/**
 * Problem: Capital City (JOI Spring Camp 2020)
 * 
 * Approach:
 * The problem asks for the minimum number of cities to merge to form a valid capital.
 * This is modeled as finding a sink Strongly Connected Component (SCC) in a dependency graph.
 * If City A's "virtual tree" (minimal connected subgraph) passes through City B, A depends on B.
 * 
 * Optimization:
 * - Use Heavy-Light Decomposition (HLD) to linearize the tree paths.
 * - Use a Segment Tree structure over the HLD array to reduce dependency edges from O(N^2) to O(N log^2 N).
 * - Graph construction is done in 2 passes (Count then Fill) to efficiently build a Compressed Sparse Row (CSR) format in parallel.
 * - Parallel sorting and unique operations are used to clean up edges.
 * - Tarjan's algorithm identifies SCCs.
 * 
 * Complexity: O(N log^2 N) time, O(N log^2 N) space.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// Fast I/O
void fast_io() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
}

// Heavy-Light Decomposition
struct HLD {
    int n;
    vector<vector<int>> adj;
    vector<int> parent, depth, heavy, head, pos, pos_to_node;
    int cur_pos;

    HLD(int n, const vector<vector<int>>& adj_in) : n(n), adj(adj_in) {
        parent.assign(n + 1, 0);
        depth.assign(n + 1, 0);
        heavy.assign(n + 1, -1);
        head.assign(n + 1, 0);
        pos.assign(n + 1, 0);
        pos_to_node.assign(n + 1, 0);
        cur_pos = 0;

        // DFS 1: Tree structure and subtree sizes
        // Iterative to prevent stack overflow
        vector<int> st;
        st.push_back(1);
        parent[1] = 0;
        depth[1] = 0;
        
        vector<int> order;
        order.reserve(n);
        
        while(!st.empty()){
            int u = st.back(); st.pop_back();
            order.push_back(u);
            for(int v : adj[u]){
                if(v != parent[u]){
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    st.push_back(v);
                }
            }
        }
        
        vector<int> sz(n + 1, 1);
        for(int i = n - 1; i >= 0; --i){
            int u = order[i];
            for(int v : adj[u]){
                if(v != parent[u]){
                    sz[u] += sz[v];
                    if(heavy[u] == -1 || sz[v] > sz[heavy[u]]){
                        heavy[u] = v;
                    }
                }
            }
        }

        // DFS 2: Decomposition
        st.clear();
        st.push_back(1);
        head[1] = 1;
        
        while(!st.empty()){
            int u = st.back(); st.pop_back();
            pos[u] = cur_pos;
            pos_to_node[cur_pos] = u;
            cur_pos++;
            
            for(int v : adj[u]){
                if(v != parent[u] && v != heavy[u]){
                    head[v] = v;
                    st.push_back(v);
                }
            }
            if(heavy[u] != -1){
                head[heavy[u]] = head[u];
                st.push_back(heavy[u]);
            }
        }
    }

    template<typename F>
    void process_path(int u, int v, F callback) {
        while (head[u] != head[v]) {
            if (depth[head[u]] > depth[head[v]]) {
                callback(pos[head[u]], pos[u] + 1); // [start, end)
                u = parent[head[u]];
            } else {
                callback(pos[head[v]], pos[v] + 1);
                v = parent[head[v]];
            }
        }
        if (depth[u] > depth[v]) swap(u, v);
        callback(pos[u], pos[v] + 1);
    }
};

// Iterative Tarjan's Algorithm
struct Tarjan {
    int n;
    const vector<int>& row_ptr;
    const vector<int>& col_ind;
    
    int scc_cnt;
    vector<int> scc_id;
    vector<int> dfn, low;
    vector<int> st;
    vector<bool> in_st;
    int timer;

    Tarjan(int n, const vector<int>& r, const vector<int>& c) 
        : n(n), row_ptr(r), col_ind(c) {
        scc_id.assign(n, -1);
        dfn.assign(n, -1);
        low.assign(n, -1);
        in_st.assign(n, false);
        st.reserve(n);
        timer = 0;
        scc_cnt = 0;
    }

    void run() {
        struct Frame {
            int u;
            int idx;
        };
        vector<Frame> stack;
        stack.reserve(n);

        for(int i = 0; i < n; ++i) {
            if(dfn[i] != -1) continue;
            
            stack.push_back({i, row_ptr[i]});
            dfn[i] = low[i] = timer++;
            st.push_back(i);
            in_st[i] = true;
            
            while(!stack.empty()) {
                Frame& fr = stack.back();
                int u = fr.u;
                int end = row_ptr[u+1];
                
                bool pushed = false;
                while(fr.idx < end) {
                    int v = col_ind[fr.idx];
                    fr.idx++;
                    
                    if(dfn[v] == -1) {
                        stack.push_back({v, row_ptr[v]});
                        dfn[v] = low[v] = timer++;
                        st.push_back(v);
                        in_st[v] = true;
                        pushed = true;
                        break;
                    } else if(in_st[v]) {
                        low[u] = min(low[u], dfn[v]);
                    }
                }
                
                if(pushed) continue;
                
                if(low[u] == dfn[u]) {
                    while(true) {
                        int v = st.back(); st.pop_back();
                        in_st[v] = false;
                        scc_id[v] = scc_cnt;
                        if(u == v) break;
                    }
                    scc_cnt++;
                }
                
                stack.pop_back();
                if(!stack.empty()) {
                    int p = stack.back().u;
                    low[p] = min(low[p], low[u]);
                }
            }
        }
    }
};

int main() {
    fast_io();

    int N, K;
    if (!(cin >> N >> K)) return 0;

    vector<vector<int>> adj(N + 1);
    for (int i = 0; i < N - 1; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> C(N + 1);
    for (int i = 1; i <= N; ++i) cin >> C[i];

    HLD hld(N, adj);

    // Group towns by city and sort by HLD pos for efficient path processing
    parlay::sequence<pair<int, int>> town_city_pairs(N);
    parlay::parallel_for(0, N, [&](size_t i) {
        town_city_pairs[i] = {C[i+1], (int)i+1};
    });
    
    auto sorted_pairs = parlay::sort(town_city_pairs, [&](const pair<int,int>& a, const pair<int,int>& b) {
        if (a.first != b.first) return a.first < b.first;
        return hld.pos[a.second] < hld.pos[b.second];
    });
    
    vector<int> city_start(K + 2, 0);
    for(const auto& p : sorted_pairs) city_start[p.first]++;
    int current = 0;
    for(int i=0; i<=K+1; ++i) {
        int cnt = city_start[i];
        city_start[i] = current;
        current += cnt;
    }

    int size = 1;
    while(size < N) size *= 2;
    int num_graph_nodes = K + 2 * size;

    // We build the graph in 2 passes:
    // Pass 1: Count out-degree of each node
    // Pass 2: Fill the edges into a CSR structure
    // Nodes: 0..K-1 (Cities), K..num_graph_nodes-1 (SegTree nodes)

    vector<int> degrees(num_graph_nodes, 0);

    // Pass 1: Count edges
    // SegTree internal nodes have 2 edges
    parlay::parallel_for(1, size, [&](size_t i) {
        degrees[K + i] = 2;
    });
    
    // SegTree leaves have 1 edge to the city they belong to
    parlay::parallel_for(0, N, [&](size_t i) {
        degrees[K + size + i] = 1;
    });
    
    // Cities have edges to SegTree ranges covering their virtual tree
    parlay::parallel_for(1, K + 1, [&](size_t k) {
        int start = city_start[k];
        int end = city_start[k+1];
        int m = end - start;
        if(m == 0) return;
        
        int cnt = 0;
        auto count_range = [&](int l, int r) {
            for (l += size, r += size; l < r; l /= 2, r /= 2) {
                if (l & 1) { cnt++; l++; }
                if (r & 1) { cnt++; r--; }
            }
        };
        
        if (m == 1) {
            int u = sorted_pairs[start].second;
            count_range(hld.pos[u], hld.pos[u] + 1);
        } else {
            for(int i=0; i<m-1; ++i) {
                int u = sorted_pairs[start+i].second;
                int v = sorted_pairs[start+i+1].second;
                hld.process_path(u, v, [&](int l, int r) {
                    if(l > r) swap(l, r);
                    count_range(l, r);
                });
            }
        }
        degrees[k-1] = cnt;
    });

    // Compute row pointers
    vector<int> row_ptr(num_graph_nodes + 1);
    row_ptr[0] = 0;
    for(int i=0; i<num_graph_nodes; ++i) {
        row_ptr[i+1] = row_ptr[i] + degrees[i];
    }
    
    int total_edges = row_ptr[num_graph_nodes];
    vector<int> col_ind(total_edges);

    // Pass 2: Fill edges
    parlay::parallel_for(1, size, [&](size_t i) {
        int idx = row_ptr[K + i];
        col_ind[idx] = K + 2*i;
        col_ind[idx+1] = K + 2*i + 1;
    });
    
    parlay::parallel_for(0, N, [&](size_t i) {
        int idx = row_ptr[K + size + i];
        int u = hld.pos_to_node[i];
        col_ind[idx] = C[u] - 1;
    });
    
    parlay::parallel_for(1, K + 1, [&](size_t k) {
        int start = city_start[k];
        int end = city_start[k+1];
        int m = end - start;
        if(m == 0) return;
        
        int idx = row_ptr[k-1];
        auto add_range = [&](int l, int r) {
            for (l += size, r += size; l < r; l /= 2, r /= 2) {
                if (l & 1) col_ind[idx++] = K + l++;
                if (r & 1) col_ind[idx++] = K + --r;
            }
        };
        
        if (m == 1) {
            int u = sorted_pairs[start].second;
            add_range(hld.pos[u], hld.pos[u] + 1);
        } else {
            for(int i=0; i<m-1; ++i) {
                int u = sorted_pairs[start+i].second;
                int v = sorted_pairs[start+i+1].second;
                hld.process_path(u, v, [&](int l, int r) {
                    if(l > r) swap(l, r);
                    add_range(l, r);
                });
            }
        }
    });

    // Pass 3: Sort and Unique edges in parallel to clean up the graph
    vector<int> new_degrees(num_graph_nodes);
    parlay::parallel_for(0, num_graph_nodes, [&](size_t i) {
        int start = row_ptr[i];
        int end = row_ptr[i+1];
        if (start < end) {
            std::sort(col_ind.begin() + start, col_ind.begin() + end);
            auto it = std::unique(col_ind.begin() + start, col_ind.begin() + end);
            new_degrees[i] = std::distance(col_ind.begin() + start, it);
        } else {
            new_degrees[i] = 0;
        }
    });
    
    vector<int> new_row_ptr(num_graph_nodes + 1);
    new_row_ptr[0] = 0;
    for(int i=0; i<num_graph_nodes; ++i) {
        new_row_ptr[i+1] = new_row_ptr[i] + new_degrees[i];
    }
    
    int new_total_edges = new_row_ptr[num_graph_nodes];
    vector<int> new_col_ind(new_total_edges);
    
    parlay::parallel_for(0, num_graph_nodes, [&](size_t i) {
        int old_start = row_ptr[i];
        int count = new_degrees[i];
        int new_start = new_row_ptr[i];
        for(int j=0; j<count; ++j) {
            new_col_ind[new_start + j] = col_ind[old_start + j];
        }
    });

    // Find SCCs
    Tarjan tarjan(num_graph_nodes, new_row_ptr, new_col_ind);
    tarjan.run();

    // Analyze SCCs
    int scc_cnt = tarjan.scc_cnt;
    vector<int> scc_size(scc_cnt, 0);
    for(int i=0; i<K; ++i) {
        if(tarjan.scc_id[i] != -1) {
            scc_size[tarjan.scc_id[i]]++;
        }
    }
    
    vector<int> scc_out_degree(scc_cnt, 0);
    for(int u=0; u<num_graph_nodes; ++u) {
        int u_scc = tarjan.scc_id[u];
        if (u_scc == -1) continue;
        int start = new_row_ptr[u];
        int end = new_row_ptr[u+1];
        for(int j=start; j<end; ++j) {
            int v = new_col_ind[j];
            int v_scc = tarjan.scc_id[v];
            if (v_scc != -1 && u_scc != v_scc) {
                scc_out_degree[u_scc] = 1;
            }
        }
    }
    
    int min_merges = K;
    for(int i=0; i<scc_cnt; ++i) {
        if(scc_out_degree[i] == 0 && scc_size[i] > 0) {
            min_merges = min(min_merges, scc_size[i] - 1);
        }
    }
    
    cout << (min_merges < 0 ? 0 : min_merges) << endl;

    return 0;
}