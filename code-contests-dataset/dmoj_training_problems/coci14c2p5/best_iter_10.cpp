/**
 * Solution for "Enchanted Forest"
 * 
 * Algorithm:
 * 1. Base DSU: Identify trees that are "always connected" (same initial height and speed).
 * 2. Event Generation: Calculate intersection times for adjacent trees.
 *    - Use parallel loops to generate events.
 *    - Filter out invalid events (past times, parallel lines, already connected).
 *    - Optimization: Use compact Event struct (16 bytes) and cross-product for comparisons.
 * 3. Sort Events: Parallel sort events by time using parlay::sort.
 * 4. Process Events: Iterate through sorted events.
 *    - Group events by identical time.
 *    - Use a temporary DSU with efficient reset (sparse set technique) to merge base components.
 *    - Update global maximum size.
 * 
 * Complexity: O(N^2 log N)
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <cmath>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// Fast I/O Buffer
const int BUFFER_SIZE = 1 << 18; // 256KB
char buffer[BUFFER_SIZE];
int buffer_pos = 0, buffer_len = 0;

inline int next_char() {
    if (buffer_pos >= buffer_len) {
        buffer_pos = 0;
        buffer_len = fread(buffer, 1, BUFFER_SIZE, stdin);
        if (buffer_len == 0) return EOF;
    }
    return buffer[buffer_pos++];
}

inline void read_int(int &x) {
    x = 0;
    int c = next_char();
    while (c <= ' ') {
        if (c == EOF) return;
        c = next_char();
    }
    while (c >= '0' && c <= '9') {
        x = x * 10 + (c - '0');
        c = next_char();
    }
}

struct Node {
    int h, v;
};

// Compact Event structure (16 bytes) to improve cache locality and sorting speed
struct Event {
    int u, v;
    int num, den; // t = num / den. Fits in int because dh <= 10^9, dv <= 10^6
};

// Standard DSU
struct DSU {
    vector<int> parent;
    vector<int> sz;
    DSU(int n) {
        parent.resize(n);
        iota(parent.begin(), parent.end(), 0);
        sz.assign(n, 1);
    }
    int find(int i) {
        // Path compression
        if (parent[i] == i) return i;
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
    read_int(N);
    
    int num_nodes = N * N;
    vector<Node> grid(num_nodes);
    for(int i=0; i<num_nodes; ++i) read_int(grid[i].h);
    for(int i=0; i<num_nodes; ++i) read_int(grid[i].v);
    
    // Phase 1: Base DSU
    // Identify components that are always connected (same h, same v)
    DSU base_dsu(num_nodes);
    
    // Horizontal edges
    for(int r=0; r<N; ++r) {
        for(int c=0; c<N-1; ++c) {
            int idx = r*N + c;
            int nxt = idx + 1;
            if (grid[idx].v == grid[nxt].v && grid[idx].h == grid[nxt].h) {
                base_dsu.unite(idx, nxt);
            }
        }
    }
    // Vertical edges
    for(int r=0; r<N-1; ++r) {
        for(int c=0; c<N; ++c) {
            int idx = r*N + c;
            int nxt = idx + N;
            if (grid[idx].v == grid[nxt].v && grid[idx].h == grid[nxt].h) {
                base_dsu.unite(idx, nxt);
            }
        }
    }
    
    vector<int> base_size(num_nodes);
    vector<int> root_map(num_nodes);
    int global_max = 0;
    
    for(int i=0; i<num_nodes; ++i) {
        int root = base_dsu.find(i);
        root_map[i] = root;
        // Only root stores the size
        if (root == i) {
            base_size[i] = base_dsu.sz[i];
            if (base_size[i] > global_max) global_max = base_size[i];
        }
    }
    
    // Phase 2: Generate Events
    int num_h_edges = N * (N - 1);
    int num_v_edges = (N - 1) * N;
    int total_edges = num_h_edges + num_v_edges;
    
    if (total_edges == 0) {
        printf("%d\n", global_max);
        return 0;
    }
    
    // We create a sequence of potential events, then filter
    parlay::sequence<Event> raw_events(total_edges);
    
    // Helper to process edge
    auto process_edge = [&](int idx, int nxt, int k) {
        long long dh = (long long)grid[nxt].h - grid[idx].h;
        long long dv = (long long)grid[idx].v - grid[nxt].v;
        
        // We want t = dh / dv >= 0
        // If dv == 0, lines are parallel. If dh == 0, always equal (handled by base DSU).
        // If dv != 0, unique intersection.
        
        if (dv == 0) {
            raw_events[k] = {-1, -1, -1, -1};
            return;
        }
        
        // Normalize signs so dv > 0
        if (dv < 0) { dh = -dh; dv = -dv; }
        
        // Check t >= 0
        if (dh < 0) {
            raw_events[k] = {-1, -1, -1, -1};
            return;
        }
        
        int root_u = root_map[idx];
        int root_v = root_map[nxt];
        
        if (root_u == root_v) {
            // Already connected in base configuration
            raw_events[k] = {-1, -1, -1, -1};
        } else {
            // Valid event
            raw_events[k] = {root_u, root_v, (int)dh, (int)dv};
        }
    };
    
    // Parallel loop for horizontal edges
    parlay::parallel_for(0, N, [&](int r) {
        for(int c=0; c<N-1; ++c) {
            int idx = r * N + c;
            process_edge(idx, idx + 1, r * (N - 1) + c);
        }
    });
    
    // Parallel loop for vertical edges
    parlay::parallel_for(0, N-1, [&](int r) {
        for(int c=0; c<N; ++c) {
            int idx = r * N + c;
            process_edge(idx, idx + N, num_h_edges + r * N + c);
        }
    });
    
    // Filter valid events
    auto events = parlay::filter(raw_events, [](const Event& e) {
        return e.u != -1;
    });
    
    // Sort events by time
    // Use cross product to compare fractions: n1/d1 < n2/d2 <=> n1*d2 < n2*d1
    events = parlay::sort(events, [](const Event& a, const Event& b) {
        long long left = (long long)a.num * b.den;
        long long right = (long long)b.num * a.den;
        return left < right;
    });
    
    // Phase 3: Process Events
    // Use a temporary DSU with efficient reset
    vector<int> temp_parent(num_nodes);
    vector<int> temp_sz(num_nodes);
    vector<int> seen_list;
    seen_list.reserve(num_nodes);
    vector<bool> seen(num_nodes, false);
    
    // Initialize temp DSU node if not seen
    auto prepare = [&](int u) {
        if (!seen[u]) {
            seen[u] = true;
            seen_list.push_back(u);
            temp_parent[u] = u;
            temp_sz[u] = base_size[u];
        }
    };
    
    // Find with path compression for temp DSU
    auto temp_find = [&](int i) {
        int root = i;
        while (temp_parent[root] != root) root = temp_parent[root];
        int curr = i;
        while (curr != root) {
            int nxt = temp_parent[curr];
            temp_parent[curr] = root;
            curr = nxt;
        }
        return root;
    };
    
    size_t n_events = events.size();
    size_t i = 0;
    while(i < n_events) {
        size_t j = i + 1;
        long long num_i = events[i].num;
        long long den_i = events[i].den;
        
        // Find end of current time batch
        while(j < n_events) {
            // Check equality: n1/d1 == n2/d2 <=> n1*d2 == n2*d1
            long long left = num_i * events[j].den;
            long long right = events[j].num * den_i;
            if (left != right) break;
            j++;
        }
        
        // Process batch
        for(size_t k = i; k < j; ++k) {
            int u = events[k].u;
            int v = events[k].v;
            
            prepare(u);
            prepare(v);
            
            int root_u = temp_find(u);
            int root_v = temp_find(v);
            
            if (root_u != root_v) {
                // Union by size
                if (temp_sz[root_u] < temp_sz[root_v]) swap(root_u, root_v);
                temp_parent[root_v] = root_u;
                temp_sz[root_u] += temp_sz[root_v];
                if (temp_sz[root_u] > global_max) global_max = temp_sz[root_u];
            }
        }
        
        // Reset only modified nodes
        for(int u : seen_list) {
            seen[u] = false;
        }
        seen_list.clear();
        
        i = j;
    }
    
    printf("%d\n", global_max);
    
    return 0;
}