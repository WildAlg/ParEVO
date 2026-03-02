/*
 * Solution for IOI '08 - Cairo, Egypt
 * 
 * Algorithm Overview:
 * 1. The problem graph is a functional graph (each node has exactly one outgoing edge).
 *    This decomposes the graph into weakly connected components, each containing exactly one cycle.
 *    The components consist of trees rooted on the cycle nodes, directed towards the cycle.
 * 2. The maximum walking distance in the park is the sum of the maximum simple paths (diameters)
 *    within each component.
 * 3. For each component, the diameter is either:
 *    a) Entirely within a tree attached to a cycle node.
 *    b) A path starting in a tree, traversing part of the cycle, and ending in another tree.
 * 4. We use Topological Sort (Peeling) to process the trees:
 *    - Compute the max depth of the tree rooted at each cycle node.
 *    - Compute the max diameter within each tree and update the component's answer.
 * 5. After peeling, the remaining nodes form the cycles. We solve the "Maximum Circular Path" problem:
 *    - Maximize depth[u] + depth[v] + dist(u, v) for distinct u, v on the cycle.
 *    - This is solved using a sliding window maximum on a linearized (doubled) cycle in O(k).
 * 6. We use the Parlay library to parallelize the initialization and cycle processing steps.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>

// Parlay library for parallelism
#include <parlay/parallel.h>
#include <parlay/sequence.h>

using namespace std;

// Optimized Fast I/O
struct FastIO {
    static const int S = 1 << 20; // 1MB buffer
    char buf[S], *p, *q;
    FILE* f;
    FastIO() : p(buf), q(buf), f(stdin) {}
    inline char gc() {
        if (p == q) {
            p = buf;
            q = buf + fread(buf, 1, S, f);
            if (p == q) return EOF;
        }
        return *p++;
    }
    inline bool readInt(int& x) {
        char c = gc();
        x = 0;
        while (c < '0' || c > '9') {
            if (c == EOF) return false;
            c = gc();
        }
        while (c >= '0' && c <= '9') {
            x = x * 10 + (c - '0');
            c = gc();
        }
        return true;
    }
} io;

const int MAXN = 1000005;

// Global arrays to store graph and state
// Using static arrays to prevent stack overflow and improve allocation speed
int adj_to[MAXN];          // Target of the bridge from i
int adj_w[MAXN];           // Length of the bridge from i
int deg[MAXN];             // In-degree for topological sort
long long depth_val[MAXN]; // Max depth of tree branch ending at node i
long long comp_ans[MAXN];  // Max diameter found within component rooted at i
int dsu_parent[MAXN];      // DSU parent for component identification

// DSU Find with Path Compression
int find_set(int v) {
    int root = v;
    while (root != dsu_parent[root]) root = dsu_parent[root];
    int curr = v;
    while (curr != root) {
        int next = dsu_parent[curr];
        dsu_parent[curr] = root;
        curr = next;
    }
    return root;
}

// DSU Union
void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) dsu_parent[b] = a;
}

// Structure to hold cycle metadata
struct CycleInfo {
    int start_idx; // Index in cycle_storage
    int length;    // Length of the cycle
    int root;      // Component root
};

int main() {
    int N;
    if (!io.readInt(N)) return 0;

    // Parallel Initialization
    parlay::parallel_for(0, N + 1, [&](int i) {
        dsu_parent[i] = i;
        deg[i] = 0;
        depth_val[i] = 0;
        comp_ans[i] = 0;
    });

    // Read Input
    for (int i = 1; i <= N; ++i) {
        int v, w;
        io.readInt(v);
        io.readInt(w);
        adj_to[i] = v;
        adj_w[i] = w;
    }

    // Build Graph State (Degrees and DSU)
    for (int i = 1; i <= N; ++i) {
        deg[adj_to[i]]++;
        union_sets(i, adj_to[i]);
    }

    // Fully compress DSU paths for O(1) access later
    parlay::parallel_for(1, N + 1, [&](int i) {
        find_set(i);
    });

    // Peeling (Topological Sort)
    // Identify initial leaves (nodes with in-degree 0)
    vector<int> q;
    q.reserve(N);
    for (int i = 1; i <= N; ++i) {
        if (deg[i] == 0) q.push_back(i);
    }

    // Process queue sequentially as dependencies exist
    int head = 0;
    while (head < q.size()) {
        int u = q[head++];
        int v = adj_to[u];
        int w = adj_w[u];
        
        // Component root
        int root = dsu_parent[v];
        
        // Update max diameter in the component passing through v.
        // It combines the current branch from u and the best previous branch entering v.
        long long path_through_v = depth_val[v] + depth_val[u] + w;
        if (path_through_v > comp_ans[root]) comp_ans[root] = path_through_v;
        
        // Update max depth for v
        if (depth_val[u] + w > depth_val[v]) depth_val[v] = depth_val[u] + w;
        
        deg[v]--;
        if (deg[v] == 0) q.push_back(v);
    }

    // Identify Cycles
    // Nodes with deg > 0 after peeling are part of a cycle.
    // We collect all cycle nodes into a flat vector to improve locality.
    vector<int> cycle_storage;
    cycle_storage.reserve(N - q.size());
    vector<CycleInfo> cycles;
    cycles.reserve((N - q.size()) / 2 + 1);

    for (int i = 1; i <= N; ++i) {
        if (deg[i] > 0) {
            int start_idx = cycle_storage.size();
            int curr = i;
            while (deg[curr] > 0) {
                deg[curr] = 0; // Mark as visited
                cycle_storage.push_back(curr);
                curr = adj_to[curr];
            }
            cycles.push_back({start_idx, (int)cycle_storage.size() - start_idx, dsu_parent[i]});
        }
    }

    // Process Cycles in Parallel
    // We compute the max path on each cycle and store it temporarily.
    vector<long long> cycle_max_paths(cycles.size());

    parlay::parallel_for(0, cycles.size(), [&](size_t i) {
        int start = cycles[i].start_idx;
        int k = cycles[i].length;
        
        if (k == 2) {
            // Optimization for cycle of size 2
            int u = cycle_storage[start];
            int v = cycle_storage[start + 1];
            long long w1 = adj_w[u];
            long long w2 = adj_w[v];
            // Path is depth[u] + depth[v] + max(edge1, edge2)
            cycle_max_paths[i] = depth_val[u] + depth_val[v] + (w1 > w2 ? w1 : w2);
            return;
        }

        // General Case: Sliding Window Maximum on Linearized Cycle
        // We double the cycle to handle wrap-around easily.
        // H[j] stores depth_val of the node
        // P[j] stores prefix sum of edge weights
        vector<long long> H(2 * k);
        vector<long long> P(2 * k + 1);
        
        P[0] = 0;
        for (int j = 0; j < k; ++j) {
            int u = cycle_storage[start + j];
            long long d = depth_val[u];
            H[j] = d;
            H[j + k] = d;
            long long w = adj_w[u];
            P[j + 1] = w;
            P[j + k + 1] = w;
        }
        
        // Compute Prefix Sums
        for (int j = 1; j <= 2 * k; ++j) {
            P[j] += P[j - 1];
        }
        
        // Sliding Window Maximum
        // We want to maximize (H[x] - P[x]) + (H[y] + P[y])
        // subject to: y - k + 1 <= x < y
        // We maintain a deque of indices x such that (H[x] - P[x]) is decreasing.
        
        vector<int> dq(2 * k);
        int head = 0, tail = 0;
        long long current_max = 0;
        
        for (int y = 0; y < 2 * k; ++y) {
            // Remove indices that are out of the window
            while (head < tail && dq[head] < y - k + 1) {
                head++;
            }
            
            // Calculate max path ending at y using the best x in the window
            if (head < tail) {
                int x = dq[head];
                long long val = (H[x] - P[x]) + (H[y] + P[y]);
                if (val > current_max) current_max = val;
            }
            
            // Add y to the deque, maintaining the monotonic property
            long long val_y = H[y] - P[y];
            while (head < tail) {
                int back = dq[tail - 1];
                if ((H[back] - P[back]) <= val_y) {
                    tail--;
                } else {
                    break;
                }
            }
            dq[tail++] = y;
        }
        cycle_max_paths[i] = current_max;
    });

    // Aggregate Results
    for (size_t i = 0; i < cycles.size(); ++i) {
        int root = cycles[i].root;
        if (cycle_max_paths[i] > comp_ans[root]) {
            comp_ans[root] = cycle_max_paths[i];
        }
    }

    // Final Summation of Component Answers
    long long total_dist = 0;
    for (int i = 1; i <= N; ++i) {
        // Only sum up for component roots
        if (dsu_parent[i] == i) {
            total_dist += comp_ans[i];
        }
    }

    cout << total_dist << endl;

    return 0;
}