/**
 * Solution for "Enchanted Forest"
 * 
 * Algorithm:
 * 1. Efficiently read input using a custom FastIO class.
 * 2. Identify "static" connections: trees that are always connected (same initial height, same speed).
 *    These are merged using a Base DSU.
 * 3. Generate "dynamic" events: adjacent trees (belonging to different base components) that 
 *    will have the same height at some time t >= 0.
 *    - Event is defined by (u, v, time).
 *    - Time is represented as a fraction num/den to maintain precision without GCD overhead.
 *    - We use the cross-product property for comparing/grouping times.
 * 4. Sort events by time using `parlay::sort_inplace`.
 * 5. Process events in time-ordered batches:
 *    - For each batch of simultaneous events, use a temporary DSU to merge base components.
 *    - Track the maximum component size.
 *    - Efficiently reset the temporary DSU by clearing only modified nodes.
 * 
 * Complexity: O(N^2 log N)
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <utility>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// ----------------------------------------------------------------------------
// Fast I/O
// ----------------------------------------------------------------------------
class FastIO {
    static const int BUF_SIZE = 1 << 18;
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

public:
    template <typename T>
    inline void read(T &x) {
        char c = get();
        x = 0;
        bool neg = false;
        while (c <= 32) {
            if (c == EOF) return;
            c = get();
        }
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

// ----------------------------------------------------------------------------
// Data Structures
// ----------------------------------------------------------------------------

// Event structure (16 bytes)
struct Event {
    int u, v;
    int num, den; // Time t = num / den. Fits in int (num <= 10^9, den <= 10^6).
};

// Base DSU for static connections
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
            int next = parent[curr];
            parent[curr] = root;
            curr = next;
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

// ----------------------------------------------------------------------------
// Main Program
// ----------------------------------------------------------------------------

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

    for (int i = 0; i < num_nodes; ++i) io.read(h[i]);
    for (int i = 0; i < num_nodes; ++i) io.read(v[i]);

    // 1. Base DSU: Merge static connections
    BaseDSU base_dsu(num_nodes);

    // Horizontal static edges
    for (int r = 0; r < N; ++r) {
        int row_offset = r * N;
        for (int c = 0; c < N - 1; ++c) {
            int idx = row_offset + c;
            int nxt = idx + 1;
            if (h[idx] == h[nxt] && v[idx] == v[nxt]) {
                base_dsu.unite(idx, nxt);
            }
        }
    }
    // Vertical static edges
    for (int r = 0; r < N - 1; ++r) {
        int row_offset = r * N;
        for (int c = 0; c < N; ++c) {
            int idx = row_offset + c;
            int nxt = idx + N;
            if (h[idx] == h[nxt] && v[idx] == v[nxt]) {
                base_dsu.unite(idx, nxt);
            }
        }
    }

    // Precompute component info
    vector<int> root_map(num_nodes);
    vector<int> base_size(num_nodes, 0);
    int global_max = 0;

    for (int i = 0; i < num_nodes; ++i) {
        int r = base_dsu.find(i);
        root_map[i] = r;
        if (base_dsu.parent[i] == i) {
            base_size[i] = base_dsu.sz[i];
            if (base_size[i] > global_max) global_max = base_size[i];
        }
    }

    // 2. Generate Events
    int num_h_edges = N * (N - 1);
    int num_v_edges = (N - 1) * N;
    int total_potential_edges = num_h_edges + num_v_edges;

    if (total_potential_edges == 0) {
        io.print(global_max);
        return 0;
    }

    auto generate_event = [&](int idx, int nxt) -> Event {
        int v1 = v[idx], v2 = v[nxt];
        int h1 = h[idx], h2 = h[nxt];
        
        long long dv = (long long)v1 - v2;
        long long dh = (long long)h2 - h1;

        if (dv == 0) return {-1, -1, -1, -1}; // Parallel

        if (dv < 0) { dv = -dv; dh = -dh; } // Normalize dv > 0

        if (dh < 0) return {-1, -1, -1, -1}; // Past

        int root_u = root_map[idx];
        int root_v = root_map[nxt];

        if (root_u == root_v) return {-1, -1, -1, -1}; // Same base component

        return {root_u, root_v, (int)dh, (int)dv};
    };

    // Parallel generation of all potential events
    auto events = parlay::tabulate(total_potential_edges, [&](int k) {
        if (k < num_h_edges) {
            int r = k / (N - 1);
            int c = k % (N - 1);
            int idx = r * N + c;
            return generate_event(idx, idx + 1);
        } else {
            int k2 = k - num_h_edges;
            int r = k2 / N;
            int c = k2 % N;
            int idx = r * N + c;
            return generate_event(idx, idx + N);
        }
    });

    // Filter valid events
    auto valid_events = parlay::filter(events, [](const Event& e) {
        return e.u != -1;
    });

    // 3. Sort Events
    parlay::sort_inplace(valid_events, [](const Event& a, const Event& b) {
        long long left = (long long)a.num * b.den;
        long long right = (long long)b.num * a.den;
        return left < right;
    });

    // 4. Process Events
    vector<int> temp_parent(num_nodes, -1);
    vector<int> temp_sz(num_nodes, 0);
    vector<int> touched;
    touched.reserve(num_nodes);

    auto touch = [&](int u) {
        if (temp_parent[u] == -1) {
            temp_parent[u] = u;
            temp_sz[u] = base_size[u];
            touched.push_back(u);
        }
    };

    auto find_temp = [&](int i) {
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

    size_t n_events = valid_events.size();
    size_t i = 0;
    while (i < n_events) {
        size_t j = i;
        long long num_i = valid_events[i].num;
        long long den_i = valid_events[i].den;

        // Group by time t (n1/d1 == n2/d2 <=> n1*d2 == n2*d1)
        while (j < n_events) {
            long long left = (long long)valid_events[j].num * den_i;
            long long right = num_i * valid_events[j].den;
            if (left != right) break;
            j++;
        }

        for (size_t k = i; k < j; ++k) {
            int u = valid_events[k].u;
            int v = valid_events[k].v;

            touch(u);
            touch(v);

            int root_u = find_temp(u);
            int root_v = find_temp(v);

            if (root_u != root_v) {
                if (temp_sz[root_u] < temp_sz[root_v]) swap(root_u, root_v);
                temp_parent[root_v] = root_u;
                temp_sz[root_u] += temp_sz[root_v];
                if (temp_sz[root_u] > global_max) global_max = temp_sz[root_u];
            }
        }

        for (int node : touched) temp_parent[node] = -1;
        touched.clear();

        i = j;
    }

    io.print(global_max);

    return 0;
}