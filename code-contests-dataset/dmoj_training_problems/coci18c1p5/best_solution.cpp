/**
 * Problem: Edge Coloring of a Bipartite Graph
 * Approach: Parallel Divide and Conquer with Iterative Cycle Extraction
 * 
 * Algorithm:
 * 1. Determine min colors needed (max degree D), set K = next_pow2(D).
 * 2. Recursive function `solve(indices, K, color_start)`:
 *    a. Base case: If m is small, switch to an optimized serial solver.
 *    b. Base case: If K=1, color all edges with `color_start`.
 *    c. Construct local graph:
 *       - Direct Mode: Used for dense/large subproblems. Uses global node indices and atomic updates.
 *       - Compressed Mode: Used for sparse subproblems. Maps active nodes to a dense range via parallel sort/unique.
 *    d. Regularize graph: Add dummy edges to odd-degree nodes and Source/Sink nodes to make the graph Eulerian (all even degrees).
 *    e. Euler Tour (Cycle Extraction): Decompose the Eulerian graph into edge-disjoint cycles.
 *       - Iteratively extract cycles to avoid recursion depth issues.
 *       - Alternating colors (0/1) along cycles ensures balanced splitting of edges at each node.
 *    f. Partition: Split edges into two sets based on the assigned 0/1 color.
 *    g. Recurse: Parallel recursive calls for the two sets with K/2 colors.
 * 
 * Optimizations:
 * - Hybrid Graph Construction: Dynamically selects the most efficient graph representation.
 * - Iterative Cycle Extraction: Replaces stack-based DFS with a cache-friendly iterative cycle finder.
 * - "Shrink K" (Serial): Skips recursion levels in the serial solver if local density drops significantly.
 * - Parallel Primitives: Uses `parlay` for efficient scanning, sorting, and partitioning.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <atomic>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// --- Data Structures ---

struct InputEdge {
    int u, v;
};

struct AdjEntry {
    int to;
    int edge_id; 
};

// Global Input Data
vector<InputEdge> g_edges;
vector<int> g_colors;
int g_L, g_R, g_M;

// --- Serial Solver ---
// Optimized for small subproblems to avoid parallel overhead

void solve_serial(const vector<int>& edge_indices, int K, int color_start) {
    size_t m = edge_indices.size();
    if (m == 0) return;

    // Coordinate Compression
    vector<int> endpoints;
    endpoints.reserve(2 * m);
    for (int idx : edge_indices) {
        endpoints.push_back(g_edges[idx].u);
        endpoints.push_back(g_edges[idx].v);
    }
    sort(endpoints.begin(), endpoints.end());
    endpoints.erase(unique(endpoints.begin(), endpoints.end()), endpoints.end());

    int num_active = endpoints.size();
    
    // Build Local Graph
    struct LocalEdge { int u, v; };
    vector<LocalEdge> local_edges(m);
    vector<int> degree(num_active, 0);

    for(size_t i=0; i<m; ++i) {
        int idx = edge_indices[i];
        int u = lower_bound(endpoints.begin(), endpoints.end(), g_edges[idx].u) - endpoints.begin();
        int v = lower_bound(endpoints.begin(), endpoints.end(), g_edges[idx].v) - endpoints.begin();
        local_edges[i] = {u, v};
        degree[u]++;
        degree[v]++;
    }

    // Optimization: Shrink K if local max degree is small
    int max_deg = 0;
    for(int d : degree) if(d > max_deg) max_deg = d;
    while(K > 1 && (K / 2) >= max_deg) {
        K /= 2;
    }

    if (K == 1) {
        for (int idx : edge_indices) {
            g_colors[idx] = color_start;
        }
        return;
    }

    // Regularize Graph (Make Eulerian)
    int S = num_active;
    int T = num_active + 1;
    int num_local_nodes = num_active + 2;
    degree.resize(num_local_nodes, 0);

    vector<int> odd_nodes;
    for(int i=0; i<num_active; ++i) {
        if(degree[i] % 2 != 0) odd_nodes.push_back(i);
    }

    int dummy_start = m;
    for(int u : odd_nodes) {
        if(endpoints[u] < g_L) { degree[u]++; degree[S]++; }
        else { degree[u]++; degree[T]++; }
    }
    if(degree[S] % 2 != 0) { degree[S]++; degree[T]++; }

    // CSR Construction
    vector<int> head(num_local_nodes + 1, 0);
    for(int i=0; i<num_local_nodes; ++i) head[i+1] = head[i] + degree[i];
    
    vector<AdjEntry> adj(head.back());
    vector<int> ptr = head; 

    for(size_t i=0; i<m; ++i) {
        int u = local_edges[i].u;
        int v = local_edges[i].v;
        adj[ptr[u]++] = {v, (int)i};
        adj[ptr[v]++] = {u, (int)i};
    }

    int current_dummy = dummy_start;
    for(int u : odd_nodes) {
        int target = (endpoints[u] < g_L) ? S : T;
        adj[ptr[u]++] = {target, current_dummy};
        adj[ptr[target]++] = {u, current_dummy};
        current_dummy++;
    }
    if(ptr[S] < head[S+1]) {
        adj[ptr[S]++] = {T, current_dummy};
        adj[ptr[T]++] = {S, current_dummy};
    }

    // Iterative Cycle Extraction
    ptr = head; // Reset ptr to start for traversal
    vector<uint8_t> edge_used(head.back()/2, 0); // Tracks usage of both real and dummy edges
    vector<uint8_t> assignment(m, 0);
    int path_len = 0;

    for (int i = 0; i < num_local_nodes; ++i) {
        // While node i has unused edges, extract a cycle passing through i
        while (ptr[i] < head[i+1]) {
            int curr = i;
            do {
                int e_id = -1;
                int next_node = -1;
                // Find next unused edge
                while (ptr[curr] < head[curr+1]) {
                    int idx = ptr[curr]++;
                    if (!edge_used[adj[idx].edge_id]) {
                        e_id = adj[idx].edge_id;
                        next_node = adj[idx].to;
                        break;
                    }
                }
                
                if (e_id == -1) break; // Should not happen in Eulerian graph

                edge_used[e_id] = 1;
                if (e_id < (int)m) {
                    assignment[e_id] = (path_len % 2);
                }
                path_len++;
                curr = next_node;
            } while (curr != i);
        }
    }

    // Partition
    vector<int> set1, set2;
    set1.reserve(m/2 + 1);
    set2.reserve(m/2 + 1);
    for(size_t i=0; i<m; ++i) {
        if(assignment[i] == 0) set1.push_back(edge_indices[i]);
        else set2.push_back(edge_indices[i]);
    }

    solve_serial(set1, K/2, color_start);
    solve_serial(set2, K/2, color_start + K/2);
}

// --- Parallel Solver ---

void solve(parlay::sequence<int> edge_indices, int K, int color_start) {
    size_t m = edge_indices.size();
    if (m == 0) return;

    // Sequential Cutoff
    if (m <= 2048) {
        vector<int> seq_indices(edge_indices.begin(), edge_indices.end());
        solve_serial(seq_indices, K, color_start);
        return;
    }

    if (K == 1) {
        parlay::parallel_for(0, m, [&](size_t i) {
            g_colors[edge_indices[i]] = color_start;
        });
        return;
    }

    // Hybrid Graph Construction
    bool use_direct = (m * 20 > (size_t)(g_L + g_R));

    int num_local_nodes;
    int S, T;
    parlay::sequence<int> head;
    parlay::sequence<AdjEntry> adj;
    parlay::sequence<int> degree;
    
    parlay::sequence<int> unique_nodes;
    parlay::sequence<pair<int, int>> local_edges;

    if (use_direct) {
        // Direct Mode: Use global indices
        int total_nodes = g_L + g_R;
        S = total_nodes; T = total_nodes + 1;
        num_local_nodes = total_nodes + 2;
        degree = parlay::sequence<int>(num_local_nodes, 0);

        parlay::parallel_for(0, m, [&](size_t i) {
            int idx = edge_indices[i];
            __sync_fetch_and_add(&degree[g_edges[idx].u], 1);
            __sync_fetch_and_add(&degree[g_edges[idx].v], 1);
        });
    } else {
        // Compressed Mode: Map to dense range
        parlay::sequence<int> endpoints(2 * m);
        parlay::parallel_for(0, m, [&](size_t i) {
            endpoints[2 * i] = g_edges[edge_indices[i]].u;
            endpoints[2 * i + 1] = g_edges[edge_indices[i]].v;
        });
        parlay::sort_inplace(endpoints);
        unique_nodes = parlay::unique(endpoints);
        size_t num_active = unique_nodes.size();
        
        S = num_active; T = num_active + 1;
        num_local_nodes = num_active + 2;
        
        local_edges = parlay::sequence<pair<int, int>>(m);
        parlay::parallel_for(0, m, [&](size_t i) {
            int idx = edge_indices[i];
            auto it_u = std::lower_bound(unique_nodes.begin(), unique_nodes.end(), g_edges[idx].u);
            auto it_v = std::lower_bound(unique_nodes.begin(), unique_nodes.end(), g_edges[idx].v);
            local_edges[i] = { (int)(it_u - unique_nodes.begin()), (int)(it_v - unique_nodes.begin()) };
        });

        degree = parlay::sequence<int>(num_local_nodes, 0);
        parlay::parallel_for(0, m, [&](size_t i) {
            __sync_fetch_and_add(&degree[local_edges[i].first], 1);
            __sync_fetch_and_add(&degree[local_edges[i].second], 1);
        });
    }

    // Regularize (Identify odd nodes)
    vector<int> odd_nodes;
    int scan_limit = use_direct ? (g_L + g_R) : unique_nodes.size();
    for(int i=0; i<scan_limit; ++i) {
        if(degree[i] % 2 != 0) {
            odd_nodes.push_back(i);
            int global_u = use_direct ? i : unique_nodes[i];
            if(global_u < g_L) { degree[i]++; degree[S]++; }
            else { degree[i]++; degree[T]++; }
        }
    }
    if(degree[S] % 2 != 0) { degree[S]++; degree[T]++; }

    // CSR Construction
    auto [prefixes, total_edges_doubled] = parlay::scan(degree);
    head = std::move(prefixes);
    head.push_back(total_edges_doubled);

    adj = parlay::sequence<AdjEntry>(total_edges_doubled);
    parlay::sequence<int> ptr = head; // Atomic counters for filling

    if (use_direct) {
        parlay::parallel_for(0, m, [&](size_t i) {
            int idx = edge_indices[i];
            int u = g_edges[idx].u;
            int v = g_edges[idx].v;
            int p_u = __sync_fetch_and_add(&ptr[u], 1);
            adj[p_u] = {v, (int)i};
            int p_v = __sync_fetch_and_add(&ptr[v], 1);
            adj[p_v] = {u, (int)i};
        });
    } else {
        parlay::parallel_for(0, m, [&](size_t i) {
            int u = local_edges[i].first;
            int v = local_edges[i].second;
            int p_u = __sync_fetch_and_add(&ptr[u], 1);
            adj[p_u] = {v, (int)i};
            int p_v = __sync_fetch_and_add(&ptr[v], 1);
            adj[p_v] = {u, (int)i};
        });
    }

    int current_dummy = m;
    for(int u : odd_nodes) {
        int global_u = use_direct ? u : unique_nodes[u];
        int target = (global_u < g_L) ? S : T;
        adj[ptr[u]++] = {target, current_dummy};
        adj[ptr[target]++] = {u, current_dummy};
        current_dummy++;
    }
    if(ptr[S] < head[S+1]) {
        adj[ptr[S]++] = {T, current_dummy};
        adj[ptr[T]++] = {S, current_dummy};
    }

    // Iterative Cycle Extraction (Serial)
    ptr = head; // Reset ptr for traversal
    
    int total_edges = total_edges_doubled / 2;
    vector<uint8_t> edge_used(total_edges, 0);
    vector<uint8_t> assignment(m, 0);
    int path_len = 0;

    for (int i = 0; i < num_local_nodes; ++i) {
        while (ptr[i] < head[i+1]) {
            int curr = i;
            do {
                int e_id = -1;
                int next_node = -1;
                while (ptr[curr] < head[curr+1]) {
                    int idx = ptr[curr]++;
                    if (!edge_used[adj[idx].edge_id]) {
                        e_id = adj[idx].edge_id;
                        next_node = adj[idx].to;
                        break;
                    }
                }
                if (e_id == -1) break;

                edge_used[e_id] = 1;
                if (e_id < (int)m) {
                    assignment[e_id] = (path_len % 2);
                }
                path_len++;
                curr = next_node;
            } while (curr != i);
        }
    }

    // Partition
    auto [offsets, count0] = parlay::scan(parlay::delayed_seq<int>(m, [&](size_t i) {
        return (assignment[i] == 0) ? 1 : 0;
    }));

    parlay::sequence<int> next_indices_1(count0);
    parlay::sequence<int> next_indices_2(m - count0);

    parlay::parallel_for(0, m, [&](size_t i) {
        if (assignment[i] == 0) next_indices_1[offsets[i]] = edge_indices[i];
        else next_indices_2[i - offsets[i]] = edge_indices[i];
    });

    parlay::par_do(
        [&]() { solve(std::move(next_indices_1), K / 2, color_start); },
        [&]() { solve(std::move(next_indices_2), K / 2, color_start + K / 2); }
    );
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> g_L >> g_R >> g_M)) return 0;

    g_edges.resize(g_M);
    g_colors.resize(g_M);
    vector<int> degree(g_L + g_R, 0);

    for (int i = 0; i < g_M; ++i) {
        int u, v;
        cin >> u >> v;
        u--; // 0-based
        v = g_L + v - 1; // 0-based, shifted to second partition
        g_edges[i] = {u, v};
        degree[u]++;
        degree[v]++;
    }

    int max_deg = 0;
    for (int d : degree) max_deg = max(max_deg, d);

    int K = 1;
    while (K < max_deg) K *= 2;

    parlay::sequence<int> initial_indices(g_M);
    parlay::parallel_for(0, g_M, [&](size_t i) {
        initial_indices[i] = i;
    });

    solve(std::move(initial_indices), K, 1);

    cout << K << "\n";
    for (int i = 0; i < g_M; ++i) {
        cout << g_colors[i] << "\n";
    }

    return 0;
}