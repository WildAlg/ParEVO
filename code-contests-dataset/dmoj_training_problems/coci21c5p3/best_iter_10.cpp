/**
 * Problem: Pinball
 * 
 * Strategy:
 * 1. Model the system with a Port Graph (4 ports per obstacle: Up, Down, Left, Right).
 * 2. Connect ports based on grid adjacency (vertical/horizontal neighbors) and internal reflection rules.
 * 3. Decompose the Port Graph into connected components.
 *    - Components are either Cycles or Paths (connected to infinity).
 *    - Validate that any Cycle has a length (number of bounces) divisible by 8.
 * 4. Build a Constraint Multigraph:
 *    - Nodes are the Cycle components found in step 3.
 *    - A special "Sink" node represents all Path components.
 *    - Edges correspond to obstacles. Each obstacle connects the two components it belongs to.
 * 5. Solve the coloring problem using Recursive Eulerian Partitioning:
 *    - The problem requires partitioning edges incident to each Cycle node into 4 equal sets.
 *    - We use recursion (depth 2) to assign 2 bits of color.
 *    - At each level, we find an Eulerian Tour (or decomposition) in the constraint graph to split edges into two balanced sets (0 and 1).
 *    - The Sink node absorbs any parity imbalances from Path components, ensuring a valid Eulerian tour exists for the connected component.
 * 
 * Implementation Details:
 * - Uses `parlay` for parallel sorting and loops.
 * - Uses an efficient array-based adjacency list for the Eulerian tour to minimize allocation overhead.
 * - Handles the Sink node logic explicitly to ensure valid tours for open paths.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// -- Data Structures --
struct Obstacle {
    int id;
    int x, y;
    char type;
};

struct Edge {
    int u, v;
    int id;
};

// -- Global Variables --
int n;
parlay::sequence<Obstacle> obstacles;
vector<int> color_bits;
vector<Edge> constraint_edges;
int sink_node;
int num_components;

// -- Helper for Adjacency List --
struct NodeEdge {
    int to;
    int next;
    int edge_idx;
};

// -- Recursive Solver --
// Assigns the 'depth'-th bit of the color to the edges in 'edge_indices'
void solve(const vector<int>& edge_indices, int depth) {
    if (depth == 2) return;
    if (edge_indices.empty()) return;

    // 1. Build Adjacency List (Array-based linked list)
    // We use local vectors to ensure thread safety during parallel recursion.
    // The graph nodes are 0 to num_components (where num_components is the Sink).
    int num_nodes = num_components + 1;
    
    // head[u] points to the index of the first edge in edges_pool for node u.
    // Initialized to -1. 
    vector<int> head(num_nodes, -1);
    
    vector<NodeEdge> edges_pool;
    edges_pool.reserve(2 * edge_indices.size());
    
    // Track active nodes to avoid iterating over all num_nodes
    vector<int> active_nodes;
    active_nodes.reserve(2 * edge_indices.size());

    for (int idx : edge_indices) {
        int u = constraint_edges[idx].u;
        int v = constraint_edges[idx].v;
        
        if (head[u] == -1) active_nodes.push_back(u);
        edges_pool.push_back({v, head[u], idx});
        head[u] = edges_pool.size() - 1;
        
        if (head[v] == -1) active_nodes.push_back(v);
        edges_pool.push_back({u, head[v], idx});
        head[v] = edges_pool.size() - 1;
    }

    // 2. Process Connected Components in the Constraint Subgraph
    // We need to track used edges for the Euler tour to avoid traversing them twice.
    // Using a vector<char> is efficient for direct access.
    vector<char> edge_used(n, 0);
    
    // Track visited nodes for component identification
    vector<bool> visited_node(num_nodes, false);

    for (int start_node : active_nodes) {
        if (visited_node[start_node]) continue;

        // Identify Component and Check for Sink
        vector<int> q;
        q.push_back(start_node);
        visited_node[start_node] = true;
        
        bool has_sink = (start_node == sink_node);
        
        int h = 0;
        while(h < (int)q.size()){
            int u = q[h++];
            if (u == sink_node) has_sink = true;
            
            for(int e = head[u]; e != -1; e = edges_pool[e].next) {
                int v = edges_pool[e].to;
                if (!visited_node[v]) {
                    visited_node[v] = true;
                    q.push_back(v);
                }
            }
        }

        // Perform Euler Tour (Hierholzer's Algorithm adapted)
        // If the component contains the Sink, we root the tour at the Sink.
        // This is crucial because Path components (which connect to Sink) may have odd degrees at the Sink,
        // effectively making the Sink the start/end of an Eulerian trail.
        int root = has_sink ? sink_node : start_node;
        
        vector<pair<int, int>> stack;
        stack.push_back({root, -1});
        
        vector<int> tour_edges;
        
        while (!stack.empty()) {
            int u = stack.back().first;
            
            // Remove already used edges from adjacency list lazily
            while (head[u] != -1) {
                int e_idx = head[u];
                if (edge_used[edges_pool[e_idx].edge_idx]) {
                    head[u] = edges_pool[e_idx].next;
                } else {
                    break;
                }
            }
            
            if (head[u] == -1) {
                // No more edges, backtrack and add edge to tour
                int inc_edge = stack.back().second;
                stack.pop_back();
                if (inc_edge != -1) {
                    tour_edges.push_back(inc_edge);
                }
            } else {
                // Traverse edge
                int e_ptr = head[u];
                int v = edges_pool[e_ptr].to;
                int idx = edges_pool[e_ptr].edge_idx;
                
                head[u] = edges_pool[e_ptr].next; // Advance head
                edge_used[idx] = 1;               // Mark used
                
                stack.push_back({v, idx});
            }
        }
        
        // Assign colors based on alternation along the tour
        // This splits the edges into two balanced sets for each node
        for (size_t i = 0; i < tour_edges.size(); i++) {
            if (i % 2 != 0) { 
                color_bits[tour_edges[i]] |= (1 << depth);
            }
        }
    }

    // 3. Recurse
    vector<int> next_0, next_1;
    next_0.reserve(edge_indices.size());
    next_1.reserve(edge_indices.size());
    
    for (int idx : edge_indices) {
        if ((color_bits[idx] >> depth) & 1) next_1.push_back(idx);
        else next_0.push_back(idx);
    }
    
    // Use parallel tasks for the next level of recursion
    parlay::par_do(
        [&]() { solve(next_0, depth + 1); },
        [&]() { solve(next_1, depth + 1); }
    );
}

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n)) return 0;

    obstacles = parlay::sequence<Obstacle>(n);
    for (int i = 0; i < n; i++) {
        obstacles[i].id = i;
        cin >> obstacles[i].x >> obstacles[i].y >> obstacles[i].type;
    }

    // --- 1. Build Port Graph ---
    // Each obstacle has 4 ports: 0:Up, 1:Down, 2:Left, 3:Right
    // Global port ID: 4*i + local_port
    parlay::sequence<int> adj(4 * n, -1);

    // Identify Vertical Connections (Up <-> Down)
    // Sort by X then Y
    auto by_x = obstacles;
    parlay::sort_inplace(by_x, [](const Obstacle& a, const Obstacle& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });

    parlay::parallel_for(0, n - 1, [&](int i) {
        if (by_x[i].x == by_x[i+1].x) {
            // by_x[i] is below by_x[i+1]. 
            // Up port (0) of i connects to Down port (1) of i+1
            int u = 4 * by_x[i].id + 0;
            int v = 4 * by_x[i+1].id + 1;
            adj[u] = v;
            adj[v] = u;
        }
    });

    // Identify Horizontal Connections (Right <-> Left)
    // Sort by Y then X
    auto by_y = obstacles;
    parlay::sort_inplace(by_y, [](const Obstacle& a, const Obstacle& b) {
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    parlay::parallel_for(0, n - 1, [&](int i) {
        if (by_y[i].y == by_y[i+1].y) {
            // by_y[i] is left of by_y[i+1].
            // Right port (3) of i connects to Left port (2) of i+1
            int u = 4 * by_y[i].id + 3;
            int v = 4 * by_y[i+1].id + 2;
            adj[u] = v;
            adj[v] = u;
        }
    });

    // Identify Internal Connections (Bounces)
    parlay::sequence<int> internal_adj(4 * n);
    parlay::parallel_for(0, n, [&](int i) {
        if (obstacles[i].type == '/') {
            // / connects Up(0)<->Left(2) and Down(1)<->Right(3)
            internal_adj[4*i+0] = 4*i+2; internal_adj[4*i+2] = 4*i+0;
            internal_adj[4*i+1] = 4*i+3; internal_adj[4*i+3] = 4*i+1;
        } else {
            // \ connects Up(0)<->Right(3) and Down(1)<->Left(2)
            internal_adj[4*i+0] = 4*i+3; internal_adj[4*i+3] = 4*i+0;
            internal_adj[4*i+1] = 4*i+2; internal_adj[4*i+2] = 4*i+1;
        }
    });

    // --- 2. Identify Cycles and Validate Lengths ---
    vector<int> comp_id(4 * n, -1);
    int c_count = 0;
    vector<bool> is_cycle;

    // Use BFS to traverse the Port Graph
    for (int i = 0; i < 4 * n; i++) {
        if (comp_id[i] == -1) {
            int c = c_count++;
            bool cycle = true;
            vector<int> q;
            q.push_back(i);
            comp_id[i] = c;
            
            int head = 0;
            while(head < (int)q.size()) {
                int u = q[head++];
                
                // Traverse Internal Edge
                int v1 = internal_adj[u];
                if (comp_id[v1] == -1) {
                    comp_id[v1] = c;
                    q.push_back(v1);
                }
                
                // Traverse External Edge
                int v2 = adj[u];
                if (v2 != -1) {
                    if (comp_id[v2] == -1) {
                        comp_id[v2] = c;
                        q.push_back(v2);
                    }
                } else {
                    cycle = false; // Hit boundary -> It's a Path, not a Cycle
                }
            }
            
            if (cycle) {
                // Cycle length in bounces = number of obstacles = number of internal edges.
                // Since each port has exactly 1 internal edge, bounces = q.size() / 2.
                // Requirement: Length must be divisible by 8.
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
    // Map component IDs to a compact range for Cycles, and map all Paths to Sink
    int cycle_idx = 0;
    vector<int> mapping(c_count);
    for(int i=0; i<c_count; ++i) {
        if(is_cycle[i]) mapping[i] = cycle_idx++;
        else mapping[i] = -1;
    }
    
    sink_node = cycle_idx;
    num_components = sink_node; // Nodes are 0..sink_node
    
    constraint_edges.reserve(n);
    for (int i = 0; i < n; i++) {
        // Each obstacle connects two internal paths.
        // Internal path 1 uses ports 0 and (2 or 3).
        // Internal path 2 uses ports 1 and (3 or 2).
        // So ports 4*i+0 and 4*i+1 always belong to different internal paths/components.
        int c1_raw = comp_id[4*i+0];
        int c2_raw = comp_id[4*i+1];
        
        int u = is_cycle[c1_raw] ? mapping[c1_raw] : sink_node;
        int v = is_cycle[c2_raw] ? mapping[c2_raw] : sink_node;
        
        constraint_edges.push_back({u, v, i});
    }

    // --- 4. Solve Coloring ---
    color_bits.assign(n, 0);
    vector<int> all_edges(n);
    for(int i=0; i<n; ++i) all_edges[i] = i;
    
    solve(all_edges, 0);

    // Output Result
    for (int i = 0; i < n; i++) {
        cout << color_bits[i] + 1 << (i == n - 1 ? "" : " ");
    }
    cout << endl;

    return 0;
}