/**
 * Problem: Pinball
 * 
 * Strategy:
 * 1. Model the pinball machine as a graph where nodes are "ports" (4 per obstacle).
 * 2. Connect ports based on:
 *    - External adjacency (ball moving between obstacles).
 *    - Internal reflection (ball bouncing within an obstacle).
 * 3. Identify connected components of ports. These correspond to ball trajectories.
 *    - A trajectory is either a Cycle or a Path (to infinity).
 *    - Constraint: For valid coloring, every Cycle must have a length (bounces) divisible by 8.
 * 4. Construct a Constraint Multigraph:
 *    - Nodes: Cycle trajectories + 1 Sink node (for all Path trajectories).
 *    - Edges: Obstacles. Each obstacle connects the two trajectories passing through it.
 * 5. Solve the coloring problem via Recursive Eulerian Partitioning:
 *    - The constraint graph has all even degrees (Cycles have length divisible by 8, Sink balances parity).
 *    - Recursively partition edges into 2 sets using Eulerian tours to assign 2 bits of color.
 *    - This ensures that for every Cycle node, the incident edges are evenly distributed among the 4 colors.
 * 
 * Implementation:
 * - Uses parlaylib for parallel sorting and processing.
 * - Efficient graph traversal and Eulerian tour construction.
 * - Handles the Sink node to manage open paths correctly.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// -- Constants and Types --
struct Obstacle {
    int id;
    int x, y;
    char type;
};

struct ConstraintEdge {
    int u, v;
    int original_idx;
};

struct NodeEdge {
    int to;
    int next;
    int edge_idx;
};

// -- Global Data --
int n_obstacles;
parlay::sequence<Obstacle> obstacles;
vector<ConstraintEdge> constraint_edges;
int sink_node;
int num_constraint_nodes;
vector<int> result_colors; // Stores the color (0-3)

// -- Recursive Solver --
// Assigns 'depth'-th bit of color to edges in 'edge_indices'
void solve(const vector<int>& edge_indices, int depth) {
    if (depth == 2) return;
    if (edge_indices.empty()) return;

    // Use local adjacency list to allow parallel execution
    // Nodes are 0 to num_constraint_nodes
    int num_nodes = num_constraint_nodes + 1;
    vector<int> head(num_nodes, -1);
    
    // Memory pool for edges in the adjacency list
    // 2 entries per undirected edge (u->v and v->u)
    vector<NodeEdge> pool;
    pool.reserve(2 * edge_indices.size());
    
    // Track active nodes to iterate efficiently
    vector<int> active_nodes;
    active_nodes.reserve(2 * edge_indices.size());

    // Build Graph
    for (int idx : edge_indices) {
        int u = constraint_edges[idx].u;
        int v = constraint_edges[idx].v;
        
        if (head[u] == -1) active_nodes.push_back(u);
        pool.push_back({v, head[u], idx});
        head[u] = pool.size() - 1;
        
        if (head[v] == -1) active_nodes.push_back(v);
        pool.push_back({u, head[v], idx});
        head[v] = pool.size() - 1;
    }

    // Process each connected component
    // Track used edges to ensure each is traversed exactly once
    // We use a global-sized vector for O(1) access.
    // Since depth is small (max 2), allocation overhead is negligible.
    vector<char> edge_used(n_obstacles, 0);
    vector<bool> visited_node(num_nodes, false);

    for (int start_node : active_nodes) {
        if (visited_node[start_node]) continue;

        // BFS to identify component and check for Sink
        // We need to know if the Sink is in this component to root the tour there
        vector<int> component_nodes;
        component_nodes.push_back(start_node);
        visited_node[start_node] = true;
        
        bool contains_sink = (start_node == sink_node);
        
        int h = 0;
        while(h < (int)component_nodes.size()){
            int u = component_nodes[h++];
            if (u == sink_node) contains_sink = true;
            
            for(int e = head[u]; e != -1; e = pool[e].next) {
                int v = pool[e].to;
                if (!visited_node[v]) {
                    visited_node[v] = true;
                    component_nodes.push_back(v);
                }
            }
        }

        // Eulerian Tour
        // If Sink is present, start/end there to handle path parity logic naturally
        int root = contains_sink ? sink_node : start_node;
        
        vector<pair<int, int>> stack;
        stack.reserve(component_nodes.size()); // Heuristic reserve
        stack.push_back({root, -1}); // Node, Incoming Edge Index
        
        vector<int> tour_edges;
        tour_edges.reserve(component_nodes.size());
        
        while (!stack.empty()) {
            int u = stack.back().first;
            
            // Find next unused edge
            while (head[u] != -1) {
                int e_ptr = head[u];
                int e_idx = pool[e_ptr].edge_idx;
                
                if (edge_used[e_idx]) {
                    head[u] = pool[e_ptr].next; // Lazy removal
                } else {
                    break;
                }
            }
            
            if (head[u] == -1) {
                // Backtrack
                int inc_edge = stack.back().second;
                stack.pop_back();
                if (inc_edge != -1) {
                    tour_edges.push_back(inc_edge);
                }
            } else {
                // Traverse
                int e_ptr = head[u];
                int v = pool[e_ptr].to;
                int idx = pool[e_ptr].edge_idx;
                
                head[u] = pool[e_ptr].next;
                edge_used[idx] = 1;
                
                stack.push_back({v, idx});
            }
        }
        
        // Color assignment based on position in tour
        // Alternating edges get different bits
        for (size_t i = 0; i < tour_edges.size(); ++i) {
            if (i % 2 != 0) {
                result_colors[tour_edges[i]] |= (1 << depth);
            }
        }
    }

    // Recursive Step
    vector<int> next_0, next_1;
    next_0.reserve(edge_indices.size() / 2 + 1);
    next_1.reserve(edge_indices.size() / 2 + 1);
    
    for (int idx : edge_indices) {
        if ((result_colors[idx] >> depth) & 1) next_1.push_back(idx);
        else next_0.push_back(idx);
    }
    
    parlay::par_do(
        [&]() { solve(next_0, depth + 1); },
        [&]() { solve(next_1, depth + 1); }
    );
}

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n_obstacles)) return 0;

    obstacles = parlay::sequence<Obstacle>(n_obstacles);
    for (int i = 0; i < n_obstacles; i++) {
        obstacles[i].id = i;
        cin >> obstacles[i].x >> obstacles[i].y >> obstacles[i].type;
    }

    // --- 1. Build Port Graph ---
    // Ports: 0:Up, 1:Down, 2:Left, 3:Right
    // Global ID: 4*obs_id + port
    parlay::sequence<int> adj(4 * n_obstacles, -1);

    // Vertical Connections: Sort by X, then Y
    auto by_x = obstacles;
    parlay::sort_inplace(by_x, [](const Obstacle& a, const Obstacle& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });

    parlay::parallel_for(0, n_obstacles - 1, [&](int i) {
        if (by_x[i].x == by_x[i+1].x) {
            // Lower obstacle Up(0) <-> Upper obstacle Down(1)
            int u = 4 * by_x[i].id + 0;
            int v = 4 * by_x[i+1].id + 1;
            adj[u] = v;
            adj[v] = u;
        }
    });

    // Horizontal Connections: Sort by Y, then X
    auto by_y = obstacles;
    parlay::sort_inplace(by_y, [](const Obstacle& a, const Obstacle& b) {
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    parlay::parallel_for(0, n_obstacles - 1, [&](int i) {
        if (by_y[i].y == by_y[i+1].y) {
            // Left obstacle Right(3) <-> Right obstacle Left(2)
            int u = 4 * by_y[i].id + 3;
            int v = 4 * by_y[i+1].id + 2;
            adj[u] = v;
            adj[v] = u;
        }
    });

    // Internal Connections (Bounces)
    // Precompute internal adjacency for fast traversal
    parlay::sequence<int> internal_adj(4 * n_obstacles);
    parlay::parallel_for(0, n_obstacles, [&](int i) {
        int base = 4 * i;
        if (obstacles[i].type == '/') {
            // / : Up(0)-Left(2), Down(1)-Right(3)
            internal_adj[base + 0] = base + 2; internal_adj[base + 2] = base + 0;
            internal_adj[base + 1] = base + 3; internal_adj[base + 3] = base + 1;
        } else {
            // \ : Up(0)-Right(3), Down(1)-Left(2)
            internal_adj[base + 0] = base + 3; internal_adj[base + 3] = base + 0;
            internal_adj[base + 1] = base + 2; internal_adj[base + 2] = base + 1;
        }
    });

    // --- 2. Identify Components (Trajectories) ---
    vector<int> comp_id(4 * n_obstacles, -1);
    int comp_count = 0;
    vector<bool> is_cycle;

    // Traverse the graph of ports to find connected components
    for (int i = 0; i < 4 * n_obstacles; i++) {
        if (comp_id[i] == -1) {
            int c = comp_count++;
            bool cycle = true;
            vector<int> q;
            q.push_back(i);
            comp_id[i] = c;
            
            int head = 0;
            while(head < (int)q.size()) {
                int u = q[head++];
                
                // Traverse Internal Edge
                int v_int = internal_adj[u];
                if (comp_id[v_int] == -1) {
                    comp_id[v_int] = c;
                    q.push_back(v_int);
                }
                
                // Traverse External Edge
                int v_ext = adj[u];
                if (v_ext != -1) {
                    if (comp_id[v_ext] == -1) {
                        comp_id[v_ext] = c;
                        q.push_back(v_ext);
                    }
                } else {
                    cycle = false; // Reached boundary -> Path
                }
            }
            
            if (cycle) {
                // Cycle length in bounces = nodes / 2 (since 2 ports per bounce)
                // Must be divisible by 8 for valid 4-coloring with even counts
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

    // --- 3. Build Constraint Multigraph ---
    int cycle_idx = 0;
    vector<int> mapping(comp_count);
    for(int i = 0; i < comp_count; ++i) {
        if (is_cycle[i]) mapping[i] = cycle_idx++;
        else mapping[i] = -1;
    }
    
    sink_node = cycle_idx;
    num_constraint_nodes = sink_node; // Valid IDs: 0 .. sink_node
    
    constraint_edges.reserve(n_obstacles);
    for (int i = 0; i < n_obstacles; i++) {
        // Obstacle i connects component of port 0 and component of port 1
        // (Port 0 and 1 always belong to different internal paths due to reflection rules)
        int c1 = comp_id[4 * i + 0];
        int c2 = comp_id[4 * i + 1];
        
        int u = is_cycle[c1] ? mapping[c1] : sink_node;
        int v = is_cycle[c2] ? mapping[c2] : sink_node;
        
        constraint_edges.push_back({u, v, i});
    }

    // --- 4. Solve ---
    result_colors.assign(n_obstacles, 0);
    vector<int> all_indices(n_obstacles);
    for(int i = 0; i < n_obstacles; ++i) all_indices[i] = i;
    
    solve(all_indices, 0);

    // --- 5. Output ---
    for (int i = 0; i < n_obstacles; i++) {
        cout << result_colors[i] + 1 << (i == n_obstacles - 1 ? "" : " ");
    }
    cout << endl;

    return 0;
}