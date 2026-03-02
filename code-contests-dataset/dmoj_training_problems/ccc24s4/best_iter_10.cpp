#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <algorithm>

#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Structure to represent an edge in the adjacency list
struct Edge {
    int to;
    int id; // Original index of the edge
};

int main() {
    // Optimize standard I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Adjacency list to store the graph
    // adj[u] contains list of {neighbor, edge_index}
    vector<vector<Edge>> adj(N + 1);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
    }

    // Result string initialized to 'G' (Grey)
    // We will overwrite characters for painted edges
    string result(M, 'G');

    // DFS state arrays
    vector<int> depth(N + 1, -1);
    vector<int> parent(N + 1, -1);
    vector<int> low(N + 1, 0);
    vector<int> preorder;
    preorder.reserve(N);

    // Iterative DFS to build the tree and compute preorder traversal
    // We iterate 1 to N to handle disconnected components
    for (int root = 1; root <= N; ++root) {
        if (depth[root] != -1) continue;

        stack<pair<int, int>> s;
        s.push({root, 0});
        depth[root] = 0;
        preorder.push_back(root);

        while (!s.empty()) {
            int u = s.top().first;
            int& idx = s.top().second; // Reference to current edge index

            if (idx < adj[u].size()) {
                int v = adj[u][idx].to;
                int id = adj[u][idx].id;
                idx++;

                if (v == parent[u]) continue;

                if (depth[v] != -1) {
                    // Back-edge: v is already visited and is an ancestor
                    // This edge remains Grey.
                    // Its existence will update 'low' values in the next step.
                } else {
                    // Tree-edge
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    preorder.push_back(v);
                    
                    // Mark as painted in result (placeholder, specific color set later)
                    result[id] = 'P'; 
                    
                    s.push({v, 0});
                }
            } else {
                s.pop();
            }
        }
    }

    // Initialize low values with depth
    for (int u : preorder) {
        low[u] = depth[u];
    }

    // Compute low values using reverse preorder (bottom-up)
    // low[u] = min depth reachable by a back-edge from subtree(u)
    for (int i = preorder.size() - 1; i >= 0; --i) {
        int u = preorder[i];
        for (const auto& e : adj[u]) {
            int v = e.to;
            if (v == parent[u]) continue;

            if (parent[v] == u) {
                // v is a child of u in the DFS tree
                low[u] = min(low[u], low[v]);
            } else {
                // (u, v) is a back-edge to an ancestor
                low[u] = min(low[u], depth[v]);
            }
        }
    }

    // Assign colors top-down
    // inc_color[u] stores the color of the edge (parent[u], u)
    // 0 for Red, 1 for Blue. Root edges default to Red.
    vector<int> inc_color(N + 1, 0); 

    for (int u : preorder) {
        int c_in = inc_color[u];
        
        for (const auto& e : adj[u]) {
            int v = e.to;
            int id = e.id;
            
            // Only process tree edges from u to children
            if (parent[v] == u) {
                int c_out;
                
                // If low[v] < depth[u], there is a back-edge from subtree(v) to an ancestor of u.
                // The path for this back-edge passes through u (parent -> u -> v).
                // We MUST alternate colors.
                if (low[v] < depth[u]) {
                    c_out = 1 - c_in;
                } else {
                    // No path passes through u via v to an ancestor.
                    // We are free to choose. Default to Red (0).
                    c_out = 0;
                }
                
                inc_color[v] = c_out;
                result[id] = (c_out == 0 ? 'R' : 'B');
            }
        }
    }

    cout << result << endl;

    return 0;
}