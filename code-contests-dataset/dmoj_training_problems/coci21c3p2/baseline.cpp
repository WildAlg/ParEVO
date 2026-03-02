#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Function to speed up standard I/O operations
void fast_io() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
}

int main() {
    fast_io();

    int n;
    long long m_in;
    // Read number of cyanobacteria (n) and connections (m)
    // Constraints: 1 <= n <= 100,000, 0 <= m < n
    if (!(cin >> n >> m_in)) return 0;
    int m = (int)m_in;

    // Build the graph using an adjacency list
    // adj[u] contains all neighbors of bacteria u
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Identify connected components in the graph
    // Each component is a tree (colony)
    vector<bool> visited(n + 1, false);
    vector<vector<int>> components;
    
    // Sequential BFS to find components
    // This is efficient enough (linear time) for N=100,000
    for (int i = 1; i <= n; ++i) {
        if (!visited[i]) {
            vector<int> component;
            queue<int> q;
            q.push(i);
            visited[i] = true;
            component.push_back(i);
            
            while (!q.empty()) {
                int u = q.front();
                q.pop();
                for (int v : adj[u]) {
                    if (!visited[v]) {
                        visited[v] = true;
                        component.push_back(v);
                        q.push(v);
                    }
                }
            }
            components.push_back(move(component));
        }
    }

    // Use parlay to calculate the diameter of each component in parallel
    // The diameter is the number of edges in the longest path within the component
    auto diameters = parlay::tabulate(components.size(), [&](size_t i) -> long long {
        auto& nodes = components[i];
        if (nodes.size() <= 1) return 0;

        // Sort nodes to enable binary search for index mapping
        // This avoids O(N) memory allocation per task by allowing O(log V_c) lookups
        sort(nodes.begin(), nodes.end());

        // Map global node index to local component index (0 to nodes.size()-1)
        auto get_local_id = [&](int global_id) {
            auto it = lower_bound(nodes.begin(), nodes.end(), global_id);
            return (int)(it - nodes.begin());
        };

        int sz = nodes.size();
        vector<int> dist(sz); // Reuse distance vector for BFS

        // BFS to find the farthest node and its distance from a start node
        auto run_bfs = [&](int start_local) -> pair<int, int> {
            fill(dist.begin(), dist.end(), -1);
            queue<int> q;
            
            dist[start_local] = 0;
            q.push(start_local);
            
            int max_d = 0;
            int farthest = start_local;
            
            while (!q.empty()) {
                int u_local = q.front();
                q.pop();
                
                if (dist[u_local] > max_d) {
                    max_d = dist[u_local];
                    farthest = u_local;
                }
                
                int u_global = nodes[u_local];
                for (int v_global : adj[u_global]) {
                    // Since it's a connected component, v_global must be in 'nodes'
                    int v_local = get_local_id(v_global);
                    if (dist[v_local] == -1) {
                        dist[v_local] = dist[u_local] + 1;
                        q.push(v_local);
                    }
                }
            }
            return {farthest, max_d};
        };

        // 2-BFS algorithm to find tree diameter:
        // 1. Find farthest node from an arbitrary node (we use local index 0)
        pair<int, int> p1 = run_bfs(0);
        // 2. Find farthest node from that node; the distance is the diameter
        pair<int, int> p2 = run_bfs(p1.first);
        
        return p2.second; // Diameter in edges
    });

    // Sum of diameters of all components
    long long total_diameter_edges = parlay::reduce(diameters);
    
    // The longest chain is formed by connecting all components in a line.
    // Each component i contributes its diameter D_i (in edges) to the path.
    // The number of nodes in that path segment is D_i + 1.
    // However, when connecting components, we merge endpoints or connect them.
    // Effectively, we sum the diameters (edges) and add the number of components.
    // Total nodes = (Sum of diameters in edges) + (Number of components)
    long long result = total_diameter_edges + components.size();
    
    cout << result << endl;

    return 0;
}