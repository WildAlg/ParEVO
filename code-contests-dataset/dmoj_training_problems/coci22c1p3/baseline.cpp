/*
 * Solution for "Little sheep Be" problem.
 * 
 * Algorithm Overview:
 * 1. Graph Construction: Nodes are spaceships, edges are touching conditions.
 *    Weights are Euclidean distances between centers.
 * 2. Component Decomposition: The graph is decomposed into connected components
 *    using a sequential BFS. This partitions the problem into independent subproblems.
 * 3. Parallel Solving (using ParlayLib):
 *    Each connected component is processed in parallel. For a component:
 *    - We express the radius of every node u as r_u = K_u * x + C_u, where x is the
 *      radius of the root node of the component. K_u is +1 or -1.
 *    - We traverse the component (BFS) to propagate these constraints.
 *    - Cycle Detection:
 *      - Even cycles (bipartite) provide consistency checks (must match distance).
 *      - Odd cycles uniquely determine the value of x.
 *    - Constraints: r_i >= 0 implies a valid range [L, R] for x.
 *    - Optimization: If x is not fixed by an odd cycle, we choose x within [L, R]
 *      to minimize the total area (sum of squared radii). The objective function
 *      is quadratic, allowing an analytical solution.
 * 4. Output: "DA" and radii if valid, "NE" otherwise.
 * 
 * Complexity: O(N + M) time and space.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <queue>

// Parlay Library Headers
#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

using namespace std;

// High precision floating point type to satisfy 10^-4 tolerance
using Real = long double;

struct Point {
    Real x, y;
};

struct Edge {
    int to;
    Real weight;
};

// Global variables to store graph and results
int n, m;
vector<Point> coords;
vector<vector<Edge>> adj;

// Global arrays for the solver.
// Since components are disjoint, parallel threads will access disjoint indices.
vector<Real> final_radii;
vector<int> K;           // Coefficient for x (1 or -1)
vector<Real> C;          // Constant term
vector<bool> visited_solve; // Visited array for the solver phase

// Calculate Euclidean distance between two spaceships
Real get_dist(int i, int j) {
    Real dx = coords[i].x - coords[j].x;
    Real dy = coords[i].y - coords[j].y;
    return sqrt(dx*dx + dy*dy);
}

// Function to solve for a single connected component
// Returns true if a valid assignment exists, false otherwise.
bool solve_component(const vector<int>& nodes) {
    if (nodes.empty()) return true;

    int root = nodes[0];
    
    // Queue for BFS: stores {current_node, parent_node}
    queue<pair<int, int>> q;
    
    // Initialize root
    q.push({root, -1});
    visited_solve[root] = true;
    K[root] = 1;
    C[root] = 0.0;

    bool x_fixed = false;
    Real fixed_val = 0.0;
    
    // BFS to propagate K and C and check cycles
    while(!q.empty()) {
        auto [u, p] = q.front();
        q.pop();
        
        for (auto& e : adj[u]) {
            int v = e.to;
            if (v == p) continue; // Skip the edge we came from
            
            Real w = e.weight;
            
            if (visited_solve[v]) {
                // Cycle detected: check consistency or determine x
                // Equation: r_u + r_v = w
                // Substitute: (K[u]*x + C[u]) + (K[v]*x + C[v]) = w
                // (K[u] + K[v])*x + (C[u] + C[v]) = w
                
                int k_sum = K[u] + K[v];
                Real c_sum = C[u] + C[v];
                
                if (k_sum == 0) {
                    // Even cycle (bipartite-like relation): 0*x + c_sum = w
                    // Must satisfy c_sum == w
                    if (abs(w - c_sum) > 1e-5) return false;
                } else {
                    // Odd cycle: +/- 2*x + c_sum = w
                    // Uniquely determines x
                    Real val = (w - c_sum) / k_sum;
                    if (x_fixed) {
                        // If x was already fixed, new value must match
                        if (abs(val - fixed_val) > 1e-5) return false;
                    } else {
                        x_fixed = true;
                        fixed_val = val;
                    }
                }
            } else {
                // Visit new node
                visited_solve[v] = true;
                K[v] = -K[u];       // K flips sign
                C[v] = w - C[u];    // C updates based on distance
                q.push({v, u});
            }
        }
    }
    
    // Determine the valid range [L, R] for x based on r_i >= 0
    // r_i = K[i]*x + C[i] >= 0
    Real L = -1e18; // Effectively -infinity
    Real R = 1e18;  // Effectively +infinity
    Real sum_kc = 0.0;
    
    for (int u : nodes) {
        if (K[u] == 1) {
            // x + C[u] >= 0  =>  x >= -C[u]
            if (-C[u] > L) L = -C[u];
        } else {
            // -x + C[u] >= 0 =>  x <= C[u]
            if (C[u] < R) R = C[u];
        }
        // Accumulate sum for optimization step
        sum_kc += K[u] * C[u];
    }
    
    // Check if a valid range exists
    if (L > R + 1e-7) return false;
    
    Real x_final;
    if (x_fixed) {
        // If x is fixed by geometry, it must be within [L, R]
        if (fixed_val < L - 1e-7 || fixed_val > R + 1e-7) return false;
        x_final = fixed_val;
    } else {
        // Optimization: Minimize sum of squared radii
        // Minimize sum((K[i]*x + C[i])^2) = sum((x + K[i]*C[i])^2)
        // Vertex of parabola is at x = -sum(K[i]*C[i]) / N
        Real x_opt = -sum_kc / nodes.size();
        
        // Clamp optimal x to valid range [L, R]
        if (x_opt < L) x_final = L;
        else if (x_opt > R) x_final = R;
        else x_final = x_opt;
    }
    
    // Compute final radii for all nodes in this component
    for (int u : nodes) {
        final_radii[u] = K[u] * x_final + C[u];
    }
    
    return true;
}

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    if (!(cin >> n >> m)) return 0;
    
    coords.resize(n + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> coords[i].x >> coords[i].y;
    }
    
    adj.resize(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        Real d = get_dist(u, v);
        adj[u].push_back({v, d});
        adj[v].push_back({u, d});
    }
    
    // Step 1: Identify connected components sequentially
    // We use a simple BFS to partition nodes into components
    vector<int> comp_map(n + 1, -1);
    vector<vector<int>> components;
    
    for (int i = 1; i <= n; ++i) {
        if (comp_map[i] == -1) {
            vector<int> comp_nodes;
            queue<int> q;
            q.push(i);
            comp_map[i] = components.size();
            comp_nodes.push_back(i);
            
            while(!q.empty()) {
                int u = q.front();
                q.pop();
                for (auto& e : adj[u]) {
                    if (comp_map[e.to] == -1) {
                        comp_map[e.to] = components.size();
                        comp_nodes.push_back(e.to);
                        q.push(e.to);
                    }
                }
            }
            components.push_back(move(comp_nodes));
        }
    }
    
    // Step 2: Prepare global structures for parallel solver
    final_radii.resize(n + 1);
    K.resize(n + 1);
    C.resize(n + 1);
    visited_solve.assign(n + 1, false);
    
    // Step 3: Process components in parallel using Parlay
    parlay::sequence<bool> results(components.size());
    
    parlay::parallel_for(0, components.size(), [&](size_t i) {
        results[i] = solve_component(components[i]);
    });
    
    // Step 4: Aggregate results
    bool possible = true;
    for (bool res : results) {
        if (!res) {
            possible = false;
            break;
        }
    }
    
    // Output
    if (possible) {
        cout << "DA\n";
        cout << fixed << setprecision(6);
        for (int i = 1; i <= n; ++i) {
            cout << final_radii[i] << "\n";
        }
    } else {
        cout << "NE\n";
    }
    
    return 0;
}