/**
 * Solution for "Enchanted Forest"
 * 
 * Algorithm:
 * 1. Efficient Input Reading: Use FastIO with buffer.
 * 2. Base Components: Identify groups of trees that are permanently connected (same h, same v).
 *    - Use a standard DSU.
 *    - Relabel components to 0..K-1 to compress the ID space.
 * 3. Event Generation (Optimized):
 *    - Use a Map-Scan-Write pattern with the parlay library.
 *    - Count valid events per row in parallel.
 *    - Compute offsets using parlay::scan (prefix sum).
 *    - Write valid events directly into the final array in parallel.
 *    - Event: Collision time t = (h2 - h1) / (v1 - v2) for adjacent trees in different components.
 * 4. Sort Events:
 *    - Sort by time using cross-product comparison (no floating point).
 *    - Use parlay::sort_inplace for high-performance parallel sorting.
 * 5. Process Events:
 *    - Group events by identical time.
 *    - Use a temporary sparse DSU to merge components.
 *    - Reset DSU efficiently using a 'touched' list to avoid clearing the whole array.
 * 
 * Complexity: O(N^2 log N)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cmath>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// Fast I/O
struct FastIO {
    static const int BUF_SIZE = 1 << 20;
    char buf[BUF_SIZE];
    int p = 0, s = 0;
    
    inline void fill() {
        s = fread(buf, 1, BUF_SIZE, stdin);
        p = 0;
    }
    
    inline char get() {
        if (p == s) fill();
        return (s == 0) ? EOF : buf[p++];
    }

    template <typename T>
    inline void read(T &x) {
        char c = get();
        x = 0;
        while (c <= 32) {
            if (c == EOF) return;
            c = get();
        }
        bool neg = false;
        if (c == '-') {
            neg = true;
            c = get();
        }
        while (c >= '0' && c <= '9') {
            x = x * 10 + (c - '0');
            c = get();
        }
        if (neg) x = -x;
    }
    
    inline void print(int x) {
        printf("%d\n", x);
    }
} io;

struct Event {
    int u, v;
    int num, den;
};

// Standard DSU for base components
struct BaseDSU {
    vector<int> parent;
    vector<int> sz;
    BaseDSU(int n) {
        parent.resize(n);
        iota(parent.begin(), parent.end(), 0);
        sz.assign(n, 1);
    }
    int find(int i) {
        int root = i;
        while (parent[root] != root) root = parent[root];
        int curr = i;
        while (curr != root) {
            int nxt = parent[curr];
            parent[curr] = root;
            curr = nxt;
        }
        return root;
    }
    void unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            if (sz[root_i] < sz[root_j]) swap(root_i, root_j);
            parent[root_j] = root_i;
            sz[root_i] += sz[root_j];
        }
    }
};

int main() {
    int N;
    io.read(N);
    
    if (N == 0) {
        io.print(0);
        return 0;
    }

    int num_nodes = N * N;
    vector<int> h(num_nodes);
    vector<int> v(num_nodes);
    for(int i=0; i<num_nodes; ++i) io.read(h[i]);
    for(int i=0; i<num_nodes; ++i) io.read(v[i]);
    
    // 1. Identify Base Components
    BaseDSU base(num_nodes);
    
    // Horizontal edges
    for(int r=0; r<N; ++r) {
        int row_off = r * N;
        for(int c=0; c<N-1; ++c) {
            int i = row_off + c;
            int j = i + 1;
            if(h[i] == h[j] && v[i] == v[j]) base.unite(i, j);
        }
    }
    // Vertical edges
    for(int r=0; r<N-1; ++r) {
        int row_off = r * N;
        for(int c=0; c<N; ++c) {
            int i = row_off + c;
            int j = i + N;
            if(h[i] == h[j] && v[i] == v[j]) base.unite(i, j);
        }
    }
    
    // Relabel components to 0..K-1
    vector<int> node_to_comp(num_nodes);
    vector<int> comp_size;
    comp_size.reserve(num_nodes);
    vector<int> root_to_comp(num_nodes, -1);
    
    int K = 0;
    int global_max = 0;
    
    for(int i=0; i<num_nodes; ++i) {
        int r = base.find(i);
        if(root_to_comp[r] == -1) {
            root_to_comp[r] = K++;
            comp_size.push_back(base.sz[r]);
            if(base.sz[r] > global_max) global_max = base.sz[r];
        }
        node_to_comp[i] = root_to_comp[r];
    }
    
    // 2. Generate Events using Map-Scan-Write
    // Map: count valid events per row
    parlay::sequence<int> row_counts(N);
    
    parlay::parallel_for(0, N, [&](int r) {
        int cnt = 0;
        int row_off = r * N;
        // Horizontal
        for(int c=0; c<N-1; ++c) {
            int i = row_off + c;
            int j = i + 1;
            int u = node_to_comp[i];
            int v_comp = node_to_comp[j];
            if (u != v_comp) {
                int dv = v[i] - v[j];
                if (dv != 0) {
                    int dh = h[j] - h[i];
                    if (dv < 0) { dv = -dv; dh = -dh; }
                    if (dh >= 0) cnt++;
                }
            }
        }
        // Vertical
        if (r < N - 1) {
            for(int c=0; c<N; ++c) {
                int i = row_off + c;
                int j = i + N;
                int u = node_to_comp[i];
                int v_comp = node_to_comp[j];
                if (u != v_comp) {
                    int dv = v[i] - v[j];
                    if (dv != 0) {
                        int dh = h[j] - h[i];
                        if (dv < 0) { dv = -dv; dh = -dh; }
                        if (dh >= 0) cnt++;
                    }
                }
            }
        }
        row_counts[r] = cnt;
    });
    
    // Scan: compute offsets
    auto [offsets, total_events] = parlay::scan(row_counts);
    
    if (total_events == 0) {
        io.print(global_max);
        return 0;
    }
    
    parlay::sequence<Event> events(total_events);
    
    // Write: fill events
    parlay::parallel_for(0, N, [&](int r) {
        int k = offsets[r];
        int row_off = r * N;
        // Horizontal
        for(int c=0; c<N-1; ++c) {
            int i = row_off + c;
            int j = i + 1;
            int u = node_to_comp[i];
            int v_comp = node_to_comp[j];
            if (u != v_comp) {
                int dv = v[i] - v[j];
                if (dv != 0) {
                    int dh = h[j] - h[i];
                    if (dv < 0) { dv = -dv; dh = -dh; }
                    if (dh >= 0) {
                        if (u > v_comp) events[k++] = {v_comp, u, dh, dv};
                        else events[k++] = {u, v_comp, dh, dv};
                    }
                }
            }
        }
        // Vertical
        if (r < N - 1) {
            for(int c=0; c<N; ++c) {
                int i = row_off + c;
                int j = i + N;
                int u = node_to_comp[i];
                int v_comp = node_to_comp[j];
                if (u != v_comp) {
                    int dv = v[i] - v[j];
                    if (dv != 0) {
                        int dh = h[j] - h[i];
                        if (dv < 0) { dv = -dv; dh = -dh; }
                        if (dh >= 0) {
                            if (u > v_comp) events[k++] = {v_comp, u, dh, dv};
                            else events[k++] = {u, v_comp, dh, dv};
                        }
                    }
                }
            }
        }
    });
    
    // Sort events by time
    parlay::sort_inplace(events, [](const Event& a, const Event& b) {
        return (long long)a.num * b.den < (long long)b.num * a.den;
    });
    
    // 3. Process Events
    // Sparse DSU structures
    vector<int> parent(K, -1);
    vector<int> sz(K);
    vector<int> touched;
    touched.reserve(K);
    
    size_t n_ev = events.size();
    size_t i = 0;
    
    while(i < n_ev) {
        size_t j = i;
        long long num_i = events[i].num;
        long long den_i = events[i].den;
        
        // Find end of current time batch
        while(j < n_ev) {
            if ((long long)events[j].num * den_i != num_i * events[j].den) break;
            j++;
        }
        
        // Process batch
        for(size_t k = i; k < j; ++k) {
            int u = events[k].u;
            int v = events[k].v;
            
            if (parent[u] == -1) {
                parent[u] = u;
                sz[u] = comp_size[u];
                touched.push_back(u);
            }
            if (parent[v] == -1) {
                parent[v] = v;
                sz[v] = comp_size[v];
                touched.push_back(v);
            }
            
            int root_u = u;
            while (parent[root_u] != root_u) root_u = parent[root_u];
            int curr = u;
            while (curr != root_u) {
                int nxt = parent[curr];
                parent[curr] = root_u;
                curr = nxt;
            }
            
            int root_v = v;
            while (parent[root_v] != root_v) root_v = parent[root_v];
            curr = v;
            while (curr != root_v) {
                int nxt = parent[curr];
                parent[curr] = root_v;
                curr = nxt;
            }
            
            if (root_u != root_v) {
                if (sz[root_u] < sz[root_v]) swap(root_u, root_v);
                parent[root_v] = root_u;
                sz[root_u] += sz[root_v];
                if (sz[root_u] > global_max) global_max = sz[root_u];
            }
        }
        
        // Reset touched nodes
        for(int node : touched) {
            parent[node] = -1;
        }
        touched.clear();
        
        i = j;
    }
    
    io.print(global_max);
    
    return 0;
}