/*
 * Solution for IOI '08 - Cairo, Egypt
 *
 * Algorithm Overview:
 * 1. Functional Graph Decomposition: The graph is a collection of components, each containing exactly one cycle.
 *    Trees are rooted on the cycle nodes and directed towards the cycle.
 * 2. Tree Peeling (Topological Sort):
 *    - We use a stack-based topological sort to process nodes with in-degree 0 (leaves).
 *    - Using a stack (LIFO) improves cache locality as we process entire branches depth-first,
 *      keeping the parent node 'v' hot in cache while updating it from its children.
 *    - For each node, we compute:
 *      a) `depth_val`: Max depth of the tree branch ending at this node.
 *      b) `max_diam`: Max diameter entirely within the tree rooted at this node.
 * 3. Cycle Processing:
 *    - After peeling, only cycle nodes have non-zero degree.
 *    - We extract cycles into contiguous "Structure of Arrays" (SoA) buffers for cache efficiency.
 *    - For each cycle, the max path is the maximum of:
 *      a) The max diameter of any tree attached to the cycle (aggregated `max_diam`).
 *      b) The max path involving cycle edges: max(depth[u] + depth[v] + dist(u, v)) for distinct u, v.
 *    - Part (b) is solved using Sliding Window Maximum on a linearized (doubled) cycle in O(k).
 * 4. Parallelism:
 *    - Parlay is used for parallel initialization and parallel processing of cycles.
 *    - Thread-local buffers are used to avoid repeated allocations during cycle processing.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>

#include <parlay/parallel.h>
#include <parlay/sequence.h>

using namespace std;

// Optimized Fast I/O
struct FastIO {
    static const int S = 1 << 20;
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

// Global Graph and State
// Static allocation for performance and to prevent stack overflow
int adj_to[MAXN];
int adj_w[MAXN];
int deg[MAXN];
long long depth_val[MAXN]; // Max depth of tree branch ending at node i
long long max_diam[MAXN];  // Max diameter found within the tree rooted at i
int q_stack[MAXN];         // Stack for topological sort

// Cycle Data (Structure of Arrays for cache efficiency)
// Stores data for all cycles contiguously
vector<long long> cycle_depths;
vector<int> cycle_weights;

struct CycleTask {
    int start_idx;         // Start index in global cycle vectors
    int length;            // Length of the cycle
    long long current_max; // Max diameter from trees attached to this cycle
};
vector<CycleTask> tasks;

// Thread-local scratch buffers for sliding window
struct TLBuffer {
    vector<long long> H;
    vector<long long> P;
    vector<int> dq;
    
    void ensure_size(int k) {
        if (H.size() < 2 * k) H.resize(2 * k);
        if (P.size() < 2 * k + 1) P.resize(2 * k + 1);
        if (dq.size() < 2 * k) dq.resize(2 * k);
    }
};

int main() {
    int N;
    if (!io.readInt(N)) return 0;

    // 1. Parallel Initialization
    parlay::parallel_for(0, N + 1, [&](int i) {
        deg[i] = 0;
        depth_val[i] = 0;
        max_diam[i] = 0;
    });

    // 2. Read Input
    for (int i = 1; i <= N; ++i) {
        int v, w;
        io.readInt(v);
        io.readInt(w);
        adj_to[i] = v;
        adj_w[i] = w;
    }

    // 3. Compute In-Degrees
    for (int i = 1; i <= N; ++i) {
        deg[adj_to[i]]++;
    }

    // 4. Peeling (Topological Sort)
    // Using a stack for DFS-like processing order (better cache locality for parent updates)
    int top = 0;
    for (int i = 1; i <= N; ++i) {
        if (deg[i] == 0) q_stack[top++] = i;
    }

    while (top > 0) {
        int u = q_stack[--top];
        int v = adj_to[u];
        int w = adj_w[u];
        
        // Update max diameter at v using u's branch.
        // This combines the current branch from u and the best previous branch entering v.
        // Since we process children sequentially, this covers all pairs of children.
        long long path_through_v = depth_val[v] + depth_val[u] + w;
        if (path_through_v > max_diam[v]) max_diam[v] = path_through_v;
        
        // Propagate max diameter found entirely within u's subtree
        if (max_diam[u] > max_diam[v]) max_diam[v] = max_diam[u];
        
        // Update max depth at v to include u's branch
        if (depth_val[u] + w > depth_val[v]) depth_val[v] = depth_val[u] + w;
        
        deg[v]--;
        if (deg[v] == 0) q_stack[top++] = v;
    }

    // 5. Extract Cycles
    cycle_depths.reserve(N);
    cycle_weights.reserve(N);
    tasks.reserve(N / 2);

    for (int i = 1; i <= N; ++i) {
        if (deg[i] > 0) {
            int start_idx = cycle_depths.size();
            long long current_cycle_max = 0;
            int curr = i;
            
            // Traverse the cycle
            while (deg[curr] > 0) {
                deg[curr] = 0; // Mark as visited
                if (max_diam[curr] > current_cycle_max) current_cycle_max = max_diam[curr];
                cycle_depths.push_back(depth_val[curr]);
                cycle_weights.push_back(adj_w[curr]);
                curr = adj_to[curr];
            }
            tasks.push_back({start_idx, (int)cycle_depths.size() - start_idx, current_cycle_max});
        }
    }

    // 6. Process Cycles in Parallel
    vector<long long> results(tasks.size());
    
    parlay::parallel_for(0, tasks.size(), [&](size_t i) {
        static thread_local TLBuffer tl;
        
        int start = tasks[i].start_idx;
        int k = tasks[i].length;
        long long best_cycle_path = 0;

        // Optimization for small cycles (k=2)
        if (k == 2) {
             long long d1 = cycle_depths[start];
             long long d2 = cycle_depths[start + 1];
             long long w1 = cycle_weights[start];
             long long w2 = cycle_weights[start + 1];
             best_cycle_path = d1 + d2 + std::max(w1, w2);
        } else {
            tl.ensure_size(k);
            long long* H = tl.H.data();
            long long* P = tl.P.data();
            int* dq = tl.dq.data();
            
            // Linearize and Double the Cycle
            // P stores prefix sums of edge weights
            P[0] = 0;
            long long current_p = 0;
            
            for (int j = 0; j < k; ++j) {
                H[j] = cycle_depths[start + j];
                H[j + k] = H[j];
                
                long long w = cycle_weights[start + j];
                current_p += w;
                P[j + 1] = current_p;
            }
            
            // Complete Prefix Sums for the doubled part
            long long total_w = current_p;
            for (int j = 1; j <= k; ++j) {
                P[k + j] = total_w + P[j];
            }
            
            // Sliding Window Maximum
            // We want to maximize (H[x] - P[x]) + (H[y] + P[y])
            // subject to: y - k + 1 <= x < y
            int head = 0, tail = 0;
            
            for (int y = 0; y < 2 * k; ++y) {
                // Remove indices that are out of the window
                while (head < tail && dq[head] <= y - k) {
                    head++;
                }
                
                // Calculate max path ending at y using the best x in the window
                if (head < tail) {
                    int x = dq[head];
                    long long val = (H[x] - P[x]) + (H[y] + P[y]);
                    if (val > best_cycle_path) best_cycle_path = val;
                }
                
                // Add y to the deque, maintaining the monotonic property
                // We want indices x with larger (H[x] - P[x]) at the front
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
        }
        
        // The answer for this component is max of cycle path and tree diameters
        results[i] = (best_cycle_path > tasks[i].current_max) ? best_cycle_path : tasks[i].current_max;
    });

    // 7. Aggregate Results
    long long total_dist = 0;
    for (long long val : results) {
        total_dist += val;
    }

    cout << total_dist << endl;

    return 0;
}