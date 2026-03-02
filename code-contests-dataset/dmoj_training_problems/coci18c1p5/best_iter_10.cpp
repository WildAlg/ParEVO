/**
 * Problem: Edge Coloring of a Bipartite Graph
 * Approach: Recursive Divide and Conquer with Euler Tour Splitting
 * Library: parlaylib for parallelism
 * 
 * Algorithm:
 * 1. Determine max degree D, set K = next_pow2(D).
 * 2. Recursive step:
 *    - If K=1, color all.
 *    - Regularize graph to be Eulerian (add dummy edges).
 *    - Find Euler tour, alternate colors 0/1.
 *    - Split edges into two sets, recurse with K/2.
 * 
 * Optimizations:
 * - Hybrid Graph Construction:
 *   - Direct Mode: Uses global indices when edge count is high (saves sorting).
 *   - Compressed Mode: Remaps nodes when edge count is low (saves memory/scan).
 * - Parallel primitives for sorting, packing.
 * - CSR graph representation.
 * - Iterative Hierholzer's algorithm.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

struct InputEdge {
    int u, v;
};

vector<InputEdge> g_edges;
vector<int> g_colors;
int g_L, g_R, g_M;

struct AdjEntry {
    int to;
    int edge_id; 
};

void solve(parlay::sequence<int> edge_indices, int K, int color_start) {
    size_t m = edge_indices.size();
    if (m == 0) return;

    if (K == 1) {
        if (m > 2000) {
            parlay::parallel_for(0, m, [&](size_t i) {
                g_colors[edge_indices[i]] = color_start;
            });
        } else {
            for (int idx : edge_indices) {
                g_colors[idx] = color_start;
            }
        }
        return;
    }

    // Heuristic: Use Direct Mode if m is large relative to N
    // This avoids O(m log m) sorting cost at top levels
    bool use_direct = (m * 20 > (size_t)(g_L + g_R));

    int num_local_nodes;
    int S, T;
    vector<int> head;
    vector<AdjEntry> adj;
    int total_edges_in_graph;

    if (use_direct) {
        // --- Direct Mode ---
        int total_nodes = g_L + g_R;
        S = total_nodes; 
        T = total_nodes + 1;
        num_local_nodes = total_nodes + 2;

        vector<int> degree(num_local_nodes, 0);
        for (size_t i = 0; i < m; ++i) {
            int idx = edge_indices[i];
            degree[g_edges[idx].u]++;
            degree[g_edges[idx].v]++;
        }

        vector<int> odd_nodes;
        // Scan all nodes to find odd degrees
        for (int i = 0; i < total_nodes; ++i) {
            if (degree[i] % 2 != 0) {
                odd_nodes.push_back(i);
                if (i < g_L) { 
                    degree[i]++;
                    degree[S]++;
                } else { 
                    degree[i]++;
                    degree[T]++;
                }
            }
        }

        if (degree[S] % 2 != 0) {
            degree[S]++;
            degree[T]++;
        }

        head.resize(num_local_nodes + 1);
        head[0] = 0;
        for (int i = 0; i < num_local_nodes; ++i) {
            head[i+1] = head[i] + degree[i];
        }

        adj.resize(head.back());
        vector<int> current_pos = head;

        for (size_t i = 0; i < m; ++i) {
            int idx = edge_indices[i];
            int u = g_edges[idx].u;
            int v = g_edges[idx].v;
            int e_id = i;
            adj[current_pos[u]++] = {v, e_id};
            adj[current_pos[v]++] = {u, e_id};
        }

        int current_dummy_id = m;
        for (int u : odd_nodes) {
            int target = (u < g_L) ? S : T;
            int e_id = current_dummy_id++;
            adj[current_pos[u]++] = {target, e_id};
            adj[current_pos[target]++] = {u, e_id};
        }

        if (current_pos[S] < head[S+1]) {
            int e_id = current_dummy_id++;
            adj[current_pos[S]++] = {T, e_id};
            adj[current_pos[T]++] = {S, e_id};
        }
        total_edges_in_graph = head.back() / 2;

    } else {
        // --- Compressed Mode ---
        parlay::sequence<int> endpoints(2 * m);
        parlay::parallel_for(0, m, [&](size_t i) {
            endpoints[2 * i] = g_edges[edge_indices[i]].u;
            endpoints[2 * i + 1] = g_edges[edge_indices[i]].v;
        });

        parlay::sort_inplace(endpoints);
        auto unique_nodes = parlay::unique(endpoints);
        size_t num_active = unique_nodes.size();

        S = num_active;
        T = num_active + 1;
        num_local_nodes = num_active + 2;

        struct LocalEdge { int u, v; };
        parlay::sequence<LocalEdge> local_edges(m);
        parlay::parallel_for(0, m, [&](size_t i) {
            int idx = edge_indices[i];
            int u_global = g_edges[idx].u;
            int v_global = g_edges[idx].v;
            
            auto it_u = std::lower_bound(unique_nodes.begin(), unique_nodes.end(), u_global);
            int u_local = std::distance(unique_nodes.begin(), it_u);
            
            auto it_v = std::lower_bound(unique_nodes.begin(), unique_nodes.end(), v_global);
            int v_local = std::distance(unique_nodes.begin(), it_v);
            
            local_edges[i] = {u_local, v_local};
        });

        vector<int> degree(num_local_nodes, 0);
        for (size_t i = 0; i < m; ++i) {
            degree[local_edges[i].u]++;
            degree[local_edges[i].v]++;
        }

        vector<int> odd_nodes;
        odd_nodes.reserve(num_active);
        for (int i = 0; i < (int)num_active; ++i) {
            if (degree[i] % 2 != 0) {
                odd_nodes.push_back(i);
                if (unique_nodes[i] < g_L) {
                    degree[i]++;
                    degree[S]++;
                } else {
                    degree[i]++;
                    degree[T]++;
                }
            }
        }

        if (degree[S] % 2 != 0) {
            degree[S]++;
            degree[T]++;
        }

        head.resize(num_local_nodes + 1);
        head[0] = 0;
        for (int i = 0; i < num_local_nodes; ++i) {
            head[i+1] = head[i] + degree[i];
        }

        adj.resize(head.back());
        vector<int> current_pos = head;

        for (size_t i = 0; i < m; ++i) {
            int u = local_edges[i].u;
            int v = local_edges[i].v;
            int e_id = i;
            adj[current_pos[u]++] = {v, e_id};
            adj[current_pos[v]++] = {u, e_id};
        }

        int current_dummy_id = m;
        for (int u : odd_nodes) {
            int target = (unique_nodes[u] < g_L) ? S : T;
            int e_id = current_dummy_id++;
            adj[current_pos[u]++] = {target, e_id};
            adj[current_pos[target]++] = {u, e_id};
        }

        if (current_pos[S] < head[S+1]) {
            int e_id = current_dummy_id++;
            adj[current_pos[S]++] = {T, e_id};
            adj[current_pos[T]++] = {S, e_id};
        }
        total_edges_in_graph = head.back() / 2;
    }

    vector<bool> edge_used(total_edges_in_graph, false);
    vector<int> ptr = head;
    vector<uint8_t> assignment(m, 0);
    
    vector<pair<int, int>> stack;
    stack.reserve(m); 

    for (int start_node = 0; start_node < num_local_nodes; ++start_node) {
        if (ptr[start_node] == head[start_node+1]) continue;

        stack.clear();
        stack.push_back({start_node, -1});
        int path_idx = 0;

        while (!stack.empty()) {
            int u = stack.back().first;
            
            int found_edge = -1;
            int found_neighbor = -1;
            
            while (ptr[u] < head[u+1]) {
                int idx = ptr[u];
                ptr[u]++; 
                
                int e_id = adj[idx].edge_id;
                if (!edge_used[e_id]) {
                    edge_used[e_id] = true;
                    found_edge = e_id;
                    found_neighbor = adj[idx].to;
                    break;
                }
            }
            
            if (found_edge != -1) {
                stack.push_back({found_neighbor, found_edge});
            } else {
                int incoming_edge = stack.back().second;
                stack.pop_back();
                
                if (incoming_edge != -1) {
                    if (incoming_edge < (int)m) {
                        assignment[incoming_edge] = (path_idx % 2);
                    }
                    path_idx++;
                }
            }
        }
    }

    auto seq1 = parlay::pack(edge_indices, parlay::delayed_seq<bool>(m, [&](size_t i) {
        return assignment[i] == 0;
    }));
    
    auto seq2 = parlay::pack(edge_indices, parlay::delayed_seq<bool>(m, [&](size_t i) {
        return assignment[i] == 1;
    }));

    parlay::par_do(
        [&]() { solve(std::move(seq1), K / 2, color_start); },
        [&]() { solve(std::move(seq2), K / 2, color_start + K / 2); }
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
        u--; 
        v = g_L + v - 1;
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