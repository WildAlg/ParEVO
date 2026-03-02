/**
 * Problem: Minimizing Merges for Capital City
 * 
 * Approach:
 * 1. Model the problem as finding a minimum set of cities to merge.
 * 2. Construct a dependency graph where an edge u -> v exists if city v must be merged
 *    when city u is chosen. This happens if a town of city v lies on a simple path
 *    between two towns of city u.
 * 3. This dependency is transitive. The problem reduces to finding a sink Strongly Connected Component (SCC)
 *    in the dependency graph with the minimum number of cities.
 * 4. To build the graph efficiently (avoiding O(N^2) edges), we use:
 *    - Heavy-Light Decomposition (HLD) to decompose paths into O(log N) intervals.
 *    - A Segment Tree style graph construction to represent range dependencies.
 *      - Edges from City u to Segment Tree Nodes covering the convex hull of towns in u.
 *      - Edges within Segment Tree (Parent -> Child).
 *      - Edges from Segment Tree Leaves to the City owning the town.
 * 5. Compute SCCs using Tarjan's algorithm.
 * 6. Identify sink SCCs (out-degree 0 in condensation graph) and find the one with min cities.
 * 
 * Complexity: O(N log N) time and O(N log N) space.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// Heavy-Light Decomposition
struct HLD {
    int n;
    vector<vector<int>> adj;
    vector<int> parent, depth, heavy, head, pos;
    int cur_pos;

    HLD(int n, const vector<vector<int>>& adj_in) : n(n), adj(adj_in) {
        parent.resize(n + 1);
        depth.resize(n + 1);
        heavy.resize(n + 1, -1);
        head.resize(n + 1);
        pos.resize(n + 1);
        cur_pos = 0;

        parent[1] = 0;
        depth[1] = 0;
        
        // Iterative DFS for subtree size and heavy child
        vector<int> order;
        order.reserve(n);
        vector<int> st;
        st.push_back(1);
        
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

        // Iterative DFS for HLD decomposition
        st.clear();
        st.push_back(1);
        head[1] = 1;
        
        while(!st.empty()){
            int u = st.back(); st.pop_back();
            pos[u] = cur_pos++;
            
            // Push non-heavy children first
            for(int v : adj[u]){
                if(v != parent[u] && v != heavy[u]){
                    head[v] = v;
                    st.push_back(v);
                }
            }
            // Push heavy child last
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
                callback(pos[head[u]], pos[u]);
                u = parent[head[u]];
            } else {
                callback(pos[head[v]], pos[v]);
                v = parent[head[v]];
            }
        }
        if (depth[u] > depth[v]) swap(u, v);
        callback(pos[u], pos[v]);
    }
};

// CSR Graph for memory efficiency
struct CSRGraph {
    int n;
    vector<int> row_ptr;
    vector<int> col_ind;
};

// Iterative Tarjan's Algorithm
struct Tarjan {
    int n;
    int scc_cnt;
    vector<int> scc_id;
    
    void run(const CSRGraph& g) {
        n = g.n;
        scc_id.assign(n, -1);
        scc_cnt = 0;
        
        vector<int> dfn(n, -1), low(n, -1);
        vector<int> st;
        vector<bool> in_st(n, false);
        int timer = 0;
        
        struct Frame {
            int u;
            int edge_idx;
        };
        vector<Frame> call_stack;
        
        for(int i = 0; i < n; ++i) {
            if(dfn[i] != -1) continue;
            
            call_stack.push_back({i, g.row_ptr[i]});
            dfn[i] = low[i] = timer++;
            st.push_back(i);
            in_st[i] = true;
            
            while(!call_stack.empty()) {
                int u = call_stack.back().u;
                int& idx = call_stack.back().edge_idx;
                
                bool pushed = false;
                int end_idx = g.row_ptr[u+1];
                
                while(idx < end_idx) {
                    int v = g.col_ind[idx];
                    idx++;
                    
                    if(dfn[v] == -1) {
                        call_stack.push_back({v, g.row_ptr[v]});
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
                
                call_stack.pop_back();
                if(!call_stack.empty()) {
                    int p = call_stack.back().u;
                    low[p] = min(low[p], low[u]);
                }
            }
        }
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

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

    // Segment Tree structure over HLD positions
    // Graph Nodes: 
    // 0..K-1: Cities
    // K..K+2*size-1: Segment Tree Nodes
    int size = 1;
    while(size < N) size *= 2;
    
    // 1. SegTree Internal Edges: Parent -> Children
    parlay::sequence<pair<int, int>> st_edges;
    if (size > 1) {
        st_edges = parlay::sequence<pair<int, int>>(2 * (size - 1));
        parlay::parallel_for(1, size, [&](size_t i) {
            st_edges[2*(i-1)] = {K + (int)i, K + 2*(int)i};
            st_edges[2*(i-1)+1] = {K + (int)i, K + 2*(int)i+1};
        });
    }
    
    // 2. SegTree Leaf -> City Edges
    parlay::sequence<pair<int, int>> leaf_edges(N);
    parlay::parallel_for(1, N + 1, [&](size_t u) {
        int p = hld.pos[u];
        int leaf_node = K + size + p;
        leaf_edges[u-1] = {leaf_node, C[u] - 1};
    });
    
    // 3. City -> SegTree Range Edges
    parlay::sequence<pair<int, int>> town_city_pairs(N);
    parlay::parallel_for(0, N, [&](size_t i) {
        town_city_pairs[i] = {C[i+1], (int)i+1};
    });
    auto sorted_towns = parlay::sort(town_city_pairs);
    
    vector<int> counts(K + 1, 0);
    for(const auto& p : sorted_towns) counts[p.first]++;
    vector<int> city_idx_start(K + 2);
    city_idx_start[1] = 0;
    for(int i=2; i<=K+1; ++i) city_idx_start[i] = city_idx_start[i-1] + counts[i-1];
    
    parlay::sequence<parlay::sequence<pair<int, int>>> city_edges_seq(K + 1);
    
    parlay::parallel_for(1, K + 1, [&](size_t k) {
        int start = city_idx_start[k];
        int end = city_idx_start[k+1];
        int m = end - start;
        if (m == 0) return;

        vector<int> nodes(m);
        for(int i=0; i<m; ++i) nodes[i] = sorted_towns[start + i].second;
        
        std::sort(nodes.begin(), nodes.end(), [&](int a, int b){
            return hld.pos[a] < hld.pos[b];
        });
        
        vector<pair<int, int>> local_edges;
        auto add_range = [&](int l, int r) {
            for (l += size, r += size; l < r; l /= 2, r /= 2) {
                if (l & 1) local_edges.push_back({(int)k - 1, K + l++});
                if (r & 1) local_edges.push_back({(int)k - 1, K + --r});
            }
        };
        
        for (int i = 0; i < m; ++i) {
            int u = nodes[i];
            int v = nodes[(i + 1) % m];
            hld.process_path(u, v, [&](int l, int r) {
                if (l > r) swap(l, r);
                add_range(l, r + 1);
            });
        }
        city_edges_seq[k] = parlay::to_sequence(local_edges);
    });
    
    auto flattened_city_edges = parlay::flatten(city_edges_seq);
    
    size_t total_edges = st_edges.size() + leaf_edges.size() + flattened_city_edges.size();
    auto all_edges = parlay::tabulate(total_edges, [&](size_t i) {
        if (i < st_edges.size()) return st_edges[i];
        i -= st_edges.size();
        if (i < leaf_edges.size()) return leaf_edges[i];
        i -= leaf_edges.size();
        return flattened_city_edges[i];
    });
    
    parlay::sort_inplace(all_edges);
    auto unique_end = std::unique(all_edges.begin(), all_edges.end());
    size_t unique_count = std::distance(all_edges.begin(), unique_end);
    
    // Build CSR Graph
    int num_nodes = K + 2 * size;
    CSRGraph g;
    g.n = num_nodes;
    g.row_ptr.assign(num_nodes + 1, 0);
    g.col_ind.resize(unique_count);
    
    for(size_t i=0; i<unique_count; ++i) {
        g.row_ptr[all_edges[i].first + 1]++;
    }
    for(int i=0; i<num_nodes; ++i) g.row_ptr[i+1] += g.row_ptr[i];
    
    parlay::parallel_for(0, unique_count, [&](size_t i) {
        g.col_ind[i] = all_edges[i].second;
    });
    
    // Run Tarjan
    Tarjan tarjan;
    tarjan.run(g);
    
    int scc_cnt = tarjan.scc_cnt;
    vector<int> scc_city_count(scc_cnt, 0);
    for(int i=0; i<K; ++i) {
        if(tarjan.scc_id[i] != -1) scc_city_count[tarjan.scc_id[i]]++;
    }
    
    vector<int> out_degree(scc_cnt, 0);
    for(size_t i=0; i<unique_count; ++i) {
        int u = all_edges[i].first;
        int v = all_edges[i].second;
        if(tarjan.scc_id[u] != tarjan.scc_id[v]) {
            out_degree[tarjan.scc_id[u]]++;
        }
    }
    
    int min_merges = K;
    for(int i=0; i<scc_cnt; ++i) {
        if(out_degree[i] == 0 && scc_city_count[i] > 0) {
            min_merges = min(min_merges, scc_city_count[i] - 1);
        }
    }
    
    cout << (min_merges < 0 ? 0 : min_merges) << endl;

    return 0;
}