/**
 * Problem: Maja's Christmas Tree
 * 
 * Solution Approach:
 * The problem asks for the maximum length of a chain formed by connecting multiple tree colonies.
 * The optimal strategy is to connect all colonies in a linear chain. For each colony, we should
 * traverse its diameter (the longest simple path). By connecting an endpoint of one colony's
 * diameter to an endpoint of the next colony's diameter, we can form a single long path whose
 * length is the sum of the diameters (in terms of nodes) of all individual colonies.
 * 
 * Algorithm:
 * 1. Parse the input graph.
 * 2. Identify connected components (colonies) using BFS.
 *    - To optimize parallel processing and cache locality, we relabel the nodes such that
 *      nodes belonging to the same component have contiguous IDs.
 * 3. Construct a Compressed Sparse Row (CSR) representation of the graph using the new IDs.
 *    - CSR is compact and provides fast adjacency iteration.
 * 4. Use the `parlay` library to process all components in parallel.
 *    - For each component, compute its diameter using the Double BFS algorithm.
 * 5. Sum the diameters of all components to get the final answer.
 * 
 * Time Complexity: O(N + M)
 * Space Complexity: O(N + M)
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

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
    if (!(cin >> n >> m_in)) return 0;
    int m = (int)m_in;

    // Temporary adjacency list for initial graph construction
    // Using 1-based indexing for input compatibility
    vector<vector<int>> initial_adj(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        initial_adj[u].push_back(v);
        initial_adj[v].push_back(u);
    }

    // Structures for relabeling nodes to group components
    // old_to_new maps 1-based original ID to 0-based new ID
    vector<int> old_to_new(n + 1, -1);
    // new_to_old maps 0-based new ID to 1-based original ID
    vector<int> new_to_old(n);
    vector<pair<int, int>> component_ranges;
    
    int current_new_id = 0;
    vector<int> q;
    q.reserve(n);
    
    // BFS to identify components and assign new contiguous IDs
    // This step ensures that nodes in the same component are stored together,
    // which significantly improves cache performance during the parallel BFS phase.
    for (int i = 1; i <= n; ++i) {
        if (old_to_new[i] == -1) {
            int start_id = current_new_id;
            
            q.clear();
            q.push_back(i);
            old_to_new[i] = current_new_id;
            new_to_old[current_new_id++] = i;
            
            int head = 0;
            while(head < q.size()) {
                int u = q[head++];
                for (int v : initial_adj[u]) {
                    if (old_to_new[v] == -1) {
                        old_to_new[v] = current_new_id;
                        new_to_old[current_new_id++] = v;
                        q.push_back(v);
                    }
                }
            }
            
            int end_id = current_new_id;
            component_ranges.push_back({start_id, end_id});
        }
    }

    // Build CSR (Compressed Sparse Row) graph using the new IDs
    // xadj: indices into csr_adj for each node (size N+1)
    vector<int> xadj(n + 1);
    // csr_adj: flattened adjacency list (size 2*M)
    vector<int> csr_adj(2 * m);
    
    // 1. Calculate offsets
    xadj[0] = 0;
    for (int i = 0; i < n; ++i) {
        int original_u = new_to_old[i];
        xadj[i+1] = xadj[i] + initial_adj[original_u].size();
    }
    
    // 2. Fill edges
    for (int i = 0; i < n; ++i) {
        int original_u = new_to_old[i];
        int current_pos = xadj[i];
        for (int original_v : initial_adj[original_u]) {
            csr_adj[current_pos++] = old_to_new[original_v];
        }
    }

    // Free initial adjacency list memory
    vector<vector<int>>().swap(initial_adj);

    // Convert ranges to parlay sequence for parallel processing
    parlay::sequence<pair<int, int>> ranges(component_ranges.begin(), component_ranges.end());

    // Calculate diameter of each component in parallel
    // We map each component range to its diameter (in nodes)
    auto diameters = parlay::map(ranges, [&](const pair<int, int>& range) -> int {
        int start = range.first;
        int end = range.second;
        int size = end - start;

        // Diameter of a single node is 1 node
        if (size <= 1) return 1;

        // BFS helper for finding farthest node and distance
        // Uses local indices [0, size-1] mapped to global [start, end-1]
        auto bfs = [&](int start_local) -> pair<int, int> {
            vector<int> dist(size, -1);
            vector<int> queue_nodes;
            queue_nodes.reserve(size);
            
            dist[start_local] = 0;
            queue_nodes.push_back(start_local);
            
            int head = 0;
            int max_dist = 0;
            int farthest_node = start_local;
            
            while(head < queue_nodes.size()) {
                int u_local = queue_nodes[head++];
                int d = dist[u_local];
                
                if (d > max_dist) {
                    max_dist = d;
                    farthest_node = u_local;
                }
                
                int u_global = start + u_local;
                // Iterate over neighbors using CSR
                for (int k = xadj[u_global]; k < xadj[u_global+1]; ++k) {
                    int v_global = csr_adj[k];
                    // v_global is guaranteed to be in [start, end)
                    int v_local = v_global - start;
                    
                    if (dist[v_local] == -1) {
                        dist[v_local] = d + 1;
                        queue_nodes.push_back(v_local);
                    }
                }
            }
            return {farthest_node, max_dist};
        };

        // Double BFS algorithm to find tree diameter:
        // 1. Find the farthest node from an arbitrary node (local index 0)
        pair<int, int> p1 = bfs(0);
        // 2. Find the farthest node from that node; the distance is the diameter
        pair<int, int> p2 = bfs(p1.first);
        
        // p2.second is the length in edges, we need number of nodes
        return p2.second + 1;
    });

    // Sum all diameters
    long long total_length = parlay::reduce(diameters);

    cout << total_length << endl;

    return 0;
}