/**
 * Problem: Capital City (JOI Spring Camp 2020)
 * 
 * Approach:
 * 1. Model the problem as finding a sink Strongly Connected Component (SCC) in a dependency graph.
 *    - Dependency: City A -> City B if the Steiner Tree of City A passes through a town of City B.
 * 
 * 2. Optimization:
 *    - Use Heavy-Light Decomposition (HLD) to linearize tree paths.
 *    - Use an Implicit Segment Tree over HLD positions to efficiently add range dependencies.
 *    - Pruning: Segment Tree leaves are mapped directly to Cities, saving N nodes and edges.
 *    - Parallelism: Use ParlayLib for sorting and graph construction.
 * 
 * 3. Algorithm:
 *    - Fast I/O and HLD construction.
 *    - Radix Sort towns by (City, HLD_Pos) using `parlay::integer_sort`.
 *    - Build CSR graph in parallel (2 passes: Calculate Degrees -> Fill Edges).
 *    - Run Iterative Tarjan's Algorithm to find SCCs.
 *    - Identify sink SCC with minimum size.
 * 
 * Complexity: O(N log^2 N) time, O(N log^2 N) space.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>
#include <parlay/io.h>

using namespace std;

// --- Fast I/O ---
struct FastIO {
    static const int S = 1 << 18;
    char buf[S], *p, *q;
    FastIO() : p(buf), q(buf) {}
    inline char gc() {
        if (p == q) {
            p = buf;
            q = buf + fread(buf, 1, S, stdin);
            if (p == q) return EOF;
        }
        return *p++;
    }
    inline void read_int(int &x) {
        x = 0;
        char c = gc();
        while (c < '0' || c > '9') c = gc();
        while (c >= '0' && c <= '9') {
            x = x * 10 + (c - '0');
            c = gc();
        }
    }
} io;

// --- Heavy-Light Decomposition ---
struct HLD {
    int n;
    vector<int> parent, depth, head, pos, pos_to_node;
    
    HLD(int n, const vector<pair<int, int>>& edges) : n(n) {
        // Build adjacency in CSR format
        vector<int> deg(n + 1, 0);
        for(const auto& e : edges) {
            deg[e.first]++;
            deg[e.second]++;
        }
        vector<int> adj_start(n + 2, 0);
        for(int i = 1; i <= n + 1; ++i) adj_start[i] = adj_start[i-1] + deg[i-1];
        
        vector<int> adj_to(adj_start[n+1]);
        vector<int> cur = adj_start;
        for(const auto& e : edges) {
            adj_to[cur[e.first]++] = e.second;
            adj_to[cur[e.second]++] = e.first;
        }

        parent.assign(n + 1, 0);
        depth.assign(n + 1, 0);
        head.assign(n + 1, 0);
        pos.assign(n + 1, 0);
        pos_to_node.assign(n + 1, 0);
        
        vector<int> heavy(n + 1, 0);
        vector<int> sz(n + 1, 1);
        
        // Iterative DFS 1: Size and Heavy Child
        vector<int> stack; stack.reserve(n);
        stack.push_back(1);
        vector<int> order; order.reserve(n);
        
        while(!stack.empty()) {
            int u = stack.back(); stack.pop_back();
            order.push_back(u);
            int start = adj_start[u];
            int end = adj_start[u+1];
            for(int i = start; i < end; ++i) {
                int v = adj_to[i];
                if(v != parent[u]) {
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    stack.push_back(v);
                }
            }
        }
        
        for(int i = n - 1; i >= 0; --i) {
            int u = order[i];
            int max_sz = -1;
            int heavy_child = 0;
            int start = adj_start[u];
            int end = adj_start[u+1];
            for(int j = start; j < end; ++j) {
                int v = adj_to[j];
                if(v != parent[u]) {
                    sz[u] += sz[v];
                    if(sz[v] > max_sz) {
                        max_sz = sz[v];
                        heavy_child = v;
                    }
                }
            }
            heavy[u] = heavy_child;
        }
        
        // Iterative DFS 2: Decomposition
        stack.clear();
        stack.push_back(1);
        head[1] = 1;
        int cur_pos = 0;
        
        while(!stack.empty()) {
            int u = stack.back(); stack.pop_back();
            pos[u] = cur_pos;
            pos_to_node[cur_pos] = u;
            cur_pos++;
            
            int h = heavy[u];
            int start = adj_start[u];
            int end = adj_start[u+1];
            
            for(int i = start; i < end; ++i) {
                int v = adj_to[i];
                if(v != parent[u] && v != h) {
                    head[v] = v;
                    stack.push_back(v);
                }
            }
            
            if(h != 0) {
                head[h] = head[u];
                stack.push_back(h);
            }
        }
    }

    template<typename F>
    inline void process_path(int u, int v, F&& callback) const {
        while (head[u] != head[v]) {
            if (depth[head[u]] > depth[head[v]]) {
                callback(pos[head[u]], pos[u] + 1);
                u = parent[head[u]];
            } else {
                callback(pos[head[v]], pos[v] + 1);
                v = parent[head[v]];
            }
        }
        if (depth[u] > depth[v]) std::swap(u, v);
        callback(pos[u], pos[v] + 1);
    }
};

// --- Iterative Tarjan's Algorithm ---
struct Tarjan {
    int n, scc_cnt, timer;
    const vector<int>& row_ptr;
    const vector<int>& col_ind;
    vector<int> scc_id, dfn, low;
    vector<char> in_st;
    vector<int> st;

    Tarjan(int n, const vector<int>& r, const vector<int>& c) 
        : n(n), row_ptr(r), col_ind(c), scc_cnt(0), timer(0) {
        scc_id.assign(n, -1);
        dfn.assign(n, -1);
        low.assign(n, -1);
        in_st.assign(n, 0);
        st.reserve(n);
    }

    void run() {
        struct Frame { int u; int idx; };
        vector<Frame> stack;
        stack.reserve(n);

        for(int i = 0; i < n; ++i) {
            if(dfn[i] != -1) continue;
            
            stack.push_back({i, row_ptr[i]});
            dfn[i] = low[i] = timer++;
            st.push_back(i);
            in_st[i] = 1;
            
            while(!stack.empty()) {
                int u = stack.back().u;
                int idx = stack.back().idx;
                int end = row_ptr[u+1];
                bool pushed = false;
                
                while(idx < end) {
                    int v = col_ind[idx];
                    
                    if(dfn[v] == -1) {
                        stack.back().idx = idx + 1; // Save progress
                        stack.push_back({v, row_ptr[v]});
                        dfn[v] = low[v] = timer++;
                        st.push_back(v);
                        in_st[v] = 1;
                        pushed = true;
                        break;
                    } else if(in_st[v]) {
                        low[u] = min(low[u], dfn[v]);
                    }
                    idx++;
                }
                
                if(pushed) continue;
                
                if(low[u] == dfn[u]) {
                    while(true) {
                        int v = st.back(); st.pop_back();
                        in_st[v] = 0;
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
    int N, K;
    io.read_int(N); io.read_int(K);

    vector<pair<int, int>> edges(N - 1);
    for(int i = 0; i < N - 1; ++i) {
        io.read_int(edges[i].first);
        io.read_int(edges[i].second);
    }

    vector<int> C(N + 1);
    for(int i = 1; i <= N; ++i) io.read_int(C[i]);

    HLD hld(N, edges);

    // Radix Sort: Key = (City << 32) | HLD_Pos
    using Key = unsigned long long;
    parlay::sequence<Key> sort_keys(N);
    parlay::parallel_for(0, N, [&](size_t i) {
        int u = i + 1;
        sort_keys[i] = ((unsigned long long)C[u] << 32) | (unsigned long long)hld.pos[u];
    });

    auto sorted_keys = parlay::integer_sort(sort_keys);

    vector<int> sorted_towns(N);
    vector<int> sorted_cities(N);
    parlay::parallel_for(0, N, [&](size_t i) {
        sorted_cities[i] = sorted_keys[i] >> 32;
        int pos = sorted_keys[i] & 0xFFFFFFFF;
        sorted_towns[i] = hld.pos_to_node[pos];
    });

    // City boundaries
    vector<int> city_start(K + 2, 0);
    for(int c : sorted_cities) city_start[c]++;
    int cur = 0;
    for(int i = 0; i <= K + 1; ++i) {
        int cnt = city_start[i];
        city_start[i] = cur;
        cur += cnt;
    }

    // Graph Construction
    int M = 1;
    while(M < N) M <<= 1;

    // Nodes: 0..K-1 (Cities), K..K+M-2 (SegTree Internal)
    // Leaves are implicit and map to Cities
    int num_st_internal = max(0, M - 1);
    int num_nodes = K + num_st_internal;
    int st_offset = K - 1; 

    // Calculate Degrees
    // 1. SegTree Internal Nodes
    vector<int> node_degrees(num_nodes, 0);
    parlay::parallel_for(1, M, [&](size_t i) {
        if (i >= M) return; 
        if (st_offset + i >= num_nodes) return; 

        int children = 0;
        int lc = 2 * i;
        if (lc < M) children++;
        else if (lc - M < N) children++;

        int rc = 2 * i + 1;
        if (rc < M) children++;
        else if (rc - M < N) children++;
        
        node_degrees[st_offset + i] = children;
    });

    // 2. City Nodes (from path queries)
    vector<int> task_counts(N, 0);
    parlay::parallel_for(0, N, [&](size_t i) {
        int city = sorted_cities[i];
        int u = sorted_towns[i];
        int v = -1;
        bool active = false;

        if (i + 1 < N && sorted_cities[i+1] == city) {
            v = sorted_towns[i+1];
            active = true;
        } else if (city_start[city+1] - city_start[city] == 1) {
            v = u;
            active = true;
        }

        if (active) {
            int cnt = 0;
            hld.process_path(u, v, [&](int l, int r) {
                for (l += M, r += M; l < r; l >>= 1, r >>= 1) {
                    if (l & 1) { cnt++; l++; }
                    if (r & 1) { cnt++; r--; }
                }
            });
            task_counts[i] = cnt;
        }
    });

    vector<int> city_extra_deg(K, 0);
    vector<int> task_offsets(N, 0);
    parlay::parallel_for(1, K + 1, [&](size_t k) {
        int start = city_start[k];
        int end = city_start[k+1];
        int sum = 0;
        for(int i = start; i < end; ++i) {
            task_offsets[i] = sum;
            sum += task_counts[i];
        }
        city_extra_deg[k-1] = sum;
    });

    parlay::parallel_for(0, K, [&](size_t i) {
        node_degrees[i] = city_extra_deg[i];
    });

    // CSR Setup
    auto [prefix, total_edges] = parlay::scan(parlay::make_slice(node_degrees));
    vector<int> row_ptr(num_nodes + 1);
    parlay::parallel_for(0, num_nodes, [&](size_t i) {
        row_ptr[i] = prefix[i];
    });
    row_ptr[num_nodes] = total_edges;

    vector<int> col_ind(total_edges);

    // Fill Edges
    // 1. SegTree Internal
    parlay::parallel_for(1, M, [&](size_t i) {
        if (st_offset + i >= num_nodes) return;
        int idx = row_ptr[st_offset + i];
        
        int lc = 2 * i;
        if (lc < M) col_ind[idx++] = st_offset + lc;
        else if (lc - M < N) col_ind[idx++] = C[hld.pos_to_node[lc - M]] - 1;

        int rc = 2 * i + 1;
        if (rc < M) col_ind[idx++] = st_offset + rc;
        else if (rc - M < N) col_ind[idx++] = C[hld.pos_to_node[rc - M]] - 1;
    });

    // 2. City Edges
    parlay::parallel_for(0, N, [&](size_t i) {
        if (task_counts[i] == 0) return;
        
        int city = sorted_cities[i];
        int u = sorted_towns[i];
        int v = -1;
        if (i + 1 < N && sorted_cities[i+1] == city) v = sorted_towns[i+1];
        else v = u;

        int base = row_ptr[city - 1] + task_offsets[i];
        
        hld.process_path(u, v, [&](int l, int r) {
            for (l += M, r += M; l < r; l >>= 1, r >>= 1) {
                if (l & 1) {
                    int node = l;
                    if (node < M) col_ind[base++] = st_offset + node;
                    else {
                        int pos = node - M;
                        if (pos < N) col_ind[base++] = C[hld.pos_to_node[pos]] - 1;
                    }
                    l++;
                }
                if (r & 1) {
                    r--;
                    int node = r;
                    if (node < M) col_ind[base++] = st_offset + node;
                    else {
                        int pos = node - M;
                        if (pos < N) col_ind[base++] = C[hld.pos_to_node[pos]] - 1;
                    }
                }
            }
        });
    });

    // Tarjan
    Tarjan tarjan(num_nodes, row_ptr, col_ind);
    tarjan.run();

    // Analyze SCCs
    int scc_cnt = tarjan.scc_cnt;
    vector<int> scc_size(scc_cnt, 0);
    for(int i = 0; i < K; ++i) {
        if(tarjan.scc_id[i] != -1) scc_size[tarjan.scc_id[i]]++;
    }

    // Check out-degrees of SCCs
    vector<char> scc_out(scc_cnt, 0);
    parlay::parallel_for(0, num_nodes, [&](size_t u) {
        int u_scc = tarjan.scc_id[u];
        if (u_scc == -1) return;
        if (scc_out[u_scc]) return;

        int start = row_ptr[u];
        int end = row_ptr[u+1];
        for(int j = start; j < end; ++j) {
            int v = col_ind[j];
            int v_scc = tarjan.scc_id[v];
            if (v_scc != -1 && u_scc != v_scc) {
                scc_out[u_scc] = 1;
                return;
            }
        }
    });

    int min_merge = K;
    for(int i = 0; i < scc_cnt; ++i) {
        if (!scc_out[i] && scc_size[i] > 0) {
            min_merge = min(min_merge, scc_size[i] - 1);
        }
    }

    cout << (min_merge < 0 ? 0 : min_merge) << endl;

    return 0;
}