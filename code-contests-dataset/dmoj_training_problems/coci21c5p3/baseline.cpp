/**
 * Problem: Pinball
 * Strategy:
 * 1. Model the system with "Ports" (4 per obstacle).
 * 2. Connect ports based on grid adjacency and internal reflection rules.
 * 3. Decompose into connected components (Cycles and Paths).
 * 4. Validate cycle lengths (must be divisible by 8).
 * 5. Build a Constraint Multigraph where nodes are Cycles (plus a Sink for Paths) and edges are Obstacles.
 * 6. Use recursive Eulerian splitting to assign colors.
 *    - Level 1: Split edges to balance sets 0 and 1.
 *    - Level 2: Split edges to balance sets 00, 01, 10, 11.
 *    - Ensure Sink node absorbs any parity imbalances.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <utility>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// Structure to represent an obstacle
struct Obstacle {
    int id;
    int x, y;
    char type;
};

// Structure for edges in the constraint graph
struct Edge {
    int u, v;
    int id;
};

// Global variables for recursion
vector<int> color_bits;
int sink_node;
vector<Edge> all_constraint_edges;

// Recursive function to split edges into balanced groups
void solve(const vector<int>& edge_indices, int depth) {
    if (depth == 2) return;

    // Determine the max node index involved to size adjacency list
    // sink_node is the largest possible index
    int max_v = sink_node; 
    vector<vector<pair<int, int>>> adj(max_v + 1);
    vector<int> nodes_in_use;
    vector<bool> active(max_v + 1, false);

    // Build adjacency list for the current subset of edges
    for (int idx : edge_indices) {
        int u = all_constraint_edges[idx].u;
        int v = all_constraint_edges[idx].v;
        adj[u].push_back({v, idx});
        adj[v].push_back({u, idx});
        if (!active[u]) { active[u] = true; nodes_in_use.push_back(u); }
        if (!active[v]) { active[v] = true; nodes_in_use.push_back(v); }
    }

    // Track used edges within this recursive step
    // We use a global-sized vector for direct indexing, but only relevant indices are accessed
    static vector<bool> local_edge_used;
    if (local_edge_used.size() < all_constraint_edges.size()) {
        local_edge_used.resize(all_constraint_edges.size(), false);
    }
    // We need to clear only the used indices for efficiency, or just use a generation counter
    // For simplicity and since we process subsets, we can just reset used ones or clear all if fast enough.
    // Given N=500k, clearing is O(N). Depth is 2. Total O(N). Safe to fill.
    for(int idx : edge_indices) local_edge_used[idx] = false;

    vector<bool> visited_node(max_v + 1, false);
    
    for (int start_node : nodes_in_use) {
        if (visited_node[start_node]) continue;
        if (adj[start_node].empty()) continue;

        // Find connected component and check if Sink is present
        vector<int> component_nodes;
        vector<int> q;
        q.push_back(start_node);
        visited_node[start_node] = true;
        
        bool has_sink = (start_node == sink_node);
        
        int head = 0;
        while(head < (int)q.size()){
            int u = q[head++];
            if (u == sink_node) has_sink = true;
            for (auto& edge : adj[u]) {
                int v = edge.first;
                if (!visited_node[v]) {
                    visited_node[v] = true;
                    q.push_back(v);
                }
            }
        }

        // Euler Tour
        // If the component contains the Sink, we MUST root the tour at the Sink
        // to ensure any parity imbalance (due to odd number of edges) is absorbed by the Sink.
        int root = has_sink ? sink_node : start_node;
        
        // Iterative Hierholzer's Algorithm / DFS
        vector<pair<int, int>> stack;
        stack.push_back({root, -1});
        
        vector<int> tour_edges;
        
        while (!stack.empty()) {
            int u = stack.back().first;
            
            // Find an unused edge
            while (!adj[u].empty()) {
                int idx = adj[u].back().second;
                if (local_edge_used[idx]) {
                    adj[u].pop_back();
                } else {
                    break;
                }
            }
            
            if (adj[u].empty()) {
                int inc_edge = stack.back().second;
                stack.pop_back();
                if (inc_edge != -1) {
                    tour_edges.push_back(inc_edge);
                }
            } else {
                auto edge = adj[u].back();
                int v = edge.first;
                int idx = edge.second;
                adj[u].pop_back();
                
                local_edge_used[idx] = true;
                stack.push_back({v, idx});
            }
        }
        
        // Assign colors based on alternation along the tour
        for (size_t i = 0; i < tour_edges.size(); i++) {
            if (i % 2 == 1) {
                color_bits[tour_edges[i]] |= (1 << depth);
            }
        }
    }

    // Prepare for next recursion level
    vector<int> next_0, next_1;
    next_0.reserve(edge_indices.size());
    next_1.reserve(edge_indices.size());
    
    for (int idx : edge_indices) {
        if ((color_bits[idx] >> depth) & 1) next_1.push_back(idx);
        else next_0.push_back(idx);
    }
    
    solve(next_0, depth + 1);
    solve(next_1, depth + 1);
}

int main() {
    // Optimize standard I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    parlay::sequence<Obstacle> obstacles(n);
    for (int i = 0; i < n; i++) {
        obstacles[i].id = i;
        cin >> obstacles[i].x >> obstacles[i].y >> obstacles[i].type;
    }

    // 1. Build Port Graph
    // Each obstacle has 4 ports: 0:Top, 1:Bottom, 2:Left, 3:Right
    // Global port ID: 4*i + local_port
    parlay::sequence<int> port_adj(4 * n, -1);

    // Sort by X then Y to link Vertical connections
    auto by_x = obstacles;
    parlay::sort_inplace(by_x, [](const Obstacle& a, const Obstacle& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });

    parlay::parallel_for(0, n - 1, [&](int i) {
        if (by_x[i].x == by_x[i+1].x) {
            // Vertical connection: Top(0) of i connects to Bottom(1) of i+1
            // (Since y_i < y_{i+1})
            int u = 4 * by_x[i].id + 0; 
            int v = 4 * by_x[i+1].id + 1;
            port_adj[u] = v;
            port_adj[v] = u;
        }
    });

    // Sort by Y then X to link Horizontal connections
    auto by_y = obstacles;
    parlay::sort_inplace(by_y, [](const Obstacle& a, const Obstacle& b) {
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    parlay::parallel_for(0, n - 1, [&](int i) {
        if (by_y[i].y == by_y[i+1].y) {
            // Horizontal connection: Right(3) of i connects to Left(2) of i+1
            // (Since x_i < x_{i+1})
            int u = 4 * by_y[i].id + 3;
            int v = 4 * by_y[i+1].id + 2;
            port_adj[u] = v;
            port_adj[v] = u;
        }
    });

    // Define internal partners for each port based on obstacle type
    // '/' : Top(0)<->Left(2), Bottom(1)<->Right(3)
    // '\' : Top(0)<->Right(3), Bottom(1)<->Left(2)
    vector<int> internal_partner(4 * n);
    parlay::parallel_for(0, n, [&](int i) {
        if (obstacles[i].type == '/') {
            internal_partner[4*i+0] = 4*i+2;
            internal_partner[4*i+2] = 4*i+0;
            internal_partner[4*i+1] = 4*i+3;
            internal_partner[4*i+3] = 4*i+1;
        } else {
            internal_partner[4*i+0] = 4*i+3;
            internal_partner[4*i+3] = 4*i+0;
            internal_partner[4*i+1] = 4*i+2;
            internal_partner[4*i+2] = 4*i+1;
        }
    });

    // 2. Identify Cycles and Paths
    vector<int> comp_id(4 * n, -1);
    int comp_count = 0;
    vector<bool> is_cycle;
    
    // Sequential traversal to identify components
    for (int i = 0; i < 4 * n; i++) {
        if (comp_id[i] == -1) {
            int c = comp_count++;
            bool cycle = true;
            
            vector<int> q;
            q.push_back(i);
            comp_id[i] = c;
            
            int head = 0;
            while(head < (int)q.size()){
                int u = q[head++];
                
                // Traverse internal edge
                int v1 = internal_partner[u];
                if (comp_id[v1] == -1) {
                    comp_id[v1] = c;
                    q.push_back(v1);
                }
                
                // Traverse external edge
                int v2 = port_adj[u];
                if (v2 != -1) {
                    if (comp_id[v2] == -1) {
                        comp_id[v2] = c;
                        q.push_back(v2);
                    }
                } else {
                    cycle = false; // Hit boundary -> Path
                }
            }
            
            // Check cycle condition: Length must be divisible by 8
            // Length = number of bounces = number of internal edges = q.size() / 2
            if (cycle) {
                if ((q.size() / 2) % 8 != 0) {
                    cout << -1 << endl;
                    return 0;
                }
                is_cycle.push_back(true);
            } else {
                is_cycle.push_back(false);
            }
        }
    }

    // 3. Build Constraint Graph
    // Remap components: Cycles get IDs 0..M-1, Paths get ID M (Sink)
    int cycles_count = 0;
    vector<int> mapping(comp_count);
    for(int i=0; i<comp_count; ++i) {
        if(is_cycle[i]) mapping[i] = cycles_count++;
        else mapping[i] = -1;
    }
    
    sink_node = cycles_count;
    vector<int> final_map(comp_count);
    for(int i=0; i<comp_count; ++i) {
        if(is_cycle[i]) final_map[i] = mapping[i];
        else final_map[i] = sink_node;
    }

    all_constraint_edges.reserve(n);
    for (int i = 0; i < n; i++) {
        // Each obstacle i connects the component of its Path 1 and Path 2
        // Path 1 ports: 4*i+0 (and its partner)
        // Path 2 ports: 4*i+1 (and its partner)
        
        int p1 = 4*i + 0;
        int p2 = 4*i + 1;
        
        int c1 = final_map[comp_id[p1]];
        int c2 = final_map[comp_id[p2]];
        
        all_constraint_edges.push_back({c1, c2, i});
    }

    // 4. Solve using Recursive Eulerian Splitting
    color_bits.assign(n, 0);
    vector<int> initial_indices(n);
    for(int i=0; i<n; ++i) initial_indices[i] = i;
    
    solve(initial_indices, 0);

    // Output result
    for (int i = 0; i < n; i++) {
        cout << color_bits[i] + 1 << (i == n - 1 ? "" : " ");
    }
    cout << endl;

    return 0;
}