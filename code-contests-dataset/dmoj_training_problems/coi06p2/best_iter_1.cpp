/*
 * Solution for Police System
 * 
 * Problem Analysis:
 * We need to answer two types of queries on a connected undirected graph:
 * 1. Is connectivity between A and B maintained after removing edge (G1, G2)?
 * 2. Is connectivity between A and B maintained after removing vertex C?
 * 
 * Approach:
 * These queries relate to bridges and articulation points.
 * - Removing an edge disconnects A and B iff the edge is a bridge and lies on the simple path 
 *   between A and B in the "Bridge Tree" (or Block-Cut Tree).
 * - Removing a vertex C disconnects A and B iff C is an articulation point and lies on the 
 *   simple path between A and B in the Block-Cut Tree.
 * 
 * Algorithm:
 * 1. Construct the Block-Cut Tree (BCT) of the graph.
 *    - BCT nodes are the original vertices and "Block" nodes representing biconnected components.
 *    - An edge exists between a vertex u and a block B if u is part of block B.
 * 2. Preprocess the BCT for LCA queries to check path inclusion efficiently.
 * 3. For Type 1 queries:
 *    - Identify the block corresponding to edge (G1, G2).
 *    - If the block consists of a single edge (it's a bridge), check if the block node is on the path between A and B in BCT.
 *    - If not a bridge, the answer is always "yes".
 * 4. For Type 2 queries:
 *    - Check if vertex C is on the path between A and B in BCT.
 * 
 * Complexity:
 * - BCT Construction: O(N + E)
 * - LCA Preprocessing: O(N log N)
 * - Query Processing: O(Q log N) (or O(Q) with O(1) LCA)
 * - Parallelism is used for sorting edges and processing queries.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
#include <string>

// Parlay library for parallelism
#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Data structures
struct EdgeMap {
    int u, v, block_id;
    // Operator for sorting/searching
    bool operator<(const EdgeMap& other) const {
        if (u != other.u) return u < other.u;
        return v < other.v;
    }
};

// Comparator for EdgeMap
bool compareEdgeMaps(const EdgeMap& a, const EdgeMap& b) {
    if (a.u != b.u) return a.u < b.u;
    return a.v < b.v;
}

// Global variables
int N, E, Q;
vector<vector<int>> adj;
vector<EdgeMap> edge_mappings;
vector<bool> is_bridge_block; // Stores if a block ID corresponds to a bridge

// Block-Cut Tree
vector<vector<int>> bct_adj;
int num_blocks = 0;

// DFS variables for BCC finding
vector<int> tin, low;
int timer;
stack<pair<int, int>> st;

// DFS to find Biconnected Components and build BCT
void dfs_bcc(int u, int p = -1) {
    tin[u] = low[u] = ++timer;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (tin[v]) {
            low[u] = min(low[u], tin[v]);
            if (tin[v] < tin[u]) {
                st.push({u, v});
            }
        } else {
            st.push({u, v});
            dfs_bcc(v, u);
            low[u] = min(low[u], low[v]);
            if (low[v] >= tin[u]) {
                // Found a block (Biconnected Component)
                int current_block_id = N + 1 + num_blocks; // Block nodes start after N
                num_blocks++;
                
                vector<int> block_vertices;
                int edge_cnt = 0;
                
                // Pop edges of the component
                while (true) {
                    pair<int, int> edge = st.top();
                    st.pop();
                    edge_cnt++;
                    
                    // Store mapping from original edge to block ID
                    int eu = min(edge.first, edge.second);
                    int ev = max(edge.first, edge.second);
                    edge_mappings.push_back({eu, ev, current_block_id});
                    
                    block_vertices.push_back(edge.first);
                    block_vertices.push_back(edge.second);
                    
                    if (edge == make_pair(u, v)) break;
                }
                
                // If a block has only 1 edge, that edge is a bridge
                is_bridge_block.push_back(edge_cnt == 1);
                
                // Add edges to Block-Cut Tree
                // Sort and unique to avoid duplicate edges to the block node
                sort(block_vertices.begin(), block_vertices.end());
                block_vertices.erase(unique(block_vertices.begin(), block_vertices.end()), block_vertices.end());
                
                for (int node : block_vertices) {
                    bct_adj[node].push_back(current_block_id);
                    bct_adj[current_block_id].push_back(node);
                }
            }
        }
    }
}

// LCA variables
const int MAXLOG = 20;
vector<vector<int>> up;
vector<int> depth;
vector<int> tin_bct, tout_bct;
int timer_bct;

// DFS for LCA preprocessing on BCT
void dfs_lca(int u, int p, int d) {
    tin_bct[u] = ++timer_bct;
    depth[u] = d;
    up[u][0] = p;
    for (int i = 1; i < MAXLOG; ++i) {
        up[u][i] = up[up[u][i-1]][i-1];
    }
    for (int v : bct_adj[u]) {
        if (v != p) {
            dfs_lca(v, u, d + 1);
        }
    }
    tout_bct[u] = ++timer_bct;
}

// Check if u is an ancestor of v
bool is_ancestor(int u, int v) {
    return tin_bct[u] <= tin_bct[v] && tout_bct[u] >= tout_bct[v];
}

// Find LCA of u and v
int get_lca(int u, int v) {
    if (is_ancestor(u, v)) return u;
    if (is_ancestor(v, u)) return v;
    for (int i = MAXLOG - 1; i >= 0; --i) {
        if (!is_ancestor(up[u][i], v)) {
            u = up[u][i];
        }
    }
    return up[u][0];
}

// Check if node x is on the simple path between u and v in BCT
bool on_path(int u, int v, int x) {
    int l = get_lca(u, v);
    // x is on path if it is a descendant of LCA(u,v) AND an ancestor of either u or v
    return is_ancestor(l, x) && (is_ancestor(x, u) || is_ancestor(x, v));
}

struct Query {
    int type;
    int A, B, C, G1, G2;
};

int main() {
    // Fast IO
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    if (!(cin >> N >> E)) return 0;
    
    adj.resize(N + 1);
    for (int i = 0; i < E; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    // Initialize DFS
    tin.assign(N + 1, 0);
    low.assign(N + 1, 0);
    timer = 0;
    
    // BCT adjacency size: N vertices + at most E blocks
    // Safe upper bound N + E + 1
    bct_adj.resize(N + E + 1);
    
    // Build Block-Cut Tree
    dfs_bcc(1);
    
    // Sort edge mappings for fast lookup using binary search
    // Using parlay::sort for parallel sorting
    parlay::sort(parlay::make_slice(edge_mappings), compareEdgeMaps);
    
    // Initialize LCA structures
    int total_nodes = N + num_blocks;
    up.assign(total_nodes + 1, vector<int>(MAXLOG));
    depth.assign(total_nodes + 1, 0);
    tin_bct.assign(total_nodes + 1, 0);
    tout_bct.assign(total_nodes + 1, 0);
    timer_bct = 0;
    
    // Preprocess LCA
    dfs_lca(1, 1, 0);
    
    cin >> Q;
    vector<Query> queries(Q);
    for (int i = 0; i < Q; ++i) {
        cin >> queries[i].type;
        if (queries[i].type == 1) {
            cin >> queries[i].A >> queries[i].B >> queries[i].G1 >> queries[i].G2;
            if (queries[i].G1 > queries[i].G2) swap(queries[i].G1, queries[i].G2);
        } else {
            cin >> queries[i].A >> queries[i].B >> queries[i].C;
        }
    }
    
    vector<string> results(Q);
    
    // Process queries in parallel
    parlay::parallel_for(0, Q, [&](size_t i) {
        const auto& q = queries[i];
        if (q.type == 1) {
            // Find the block ID for edge (G1, G2)
            EdgeMap target = {q.G1, q.G2, 0};
            auto it = lower_bound(edge_mappings.begin(), edge_mappings.end(), target, compareEdgeMaps);
            
            if (it != edge_mappings.end() && it->u == q.G1 && it->v == q.G2) {
                int bid = it->block_id;
                int b_idx = bid - (N + 1);
                
                // If the block is a bridge (single edge), check if it's on the path
                if (is_bridge_block[b_idx]) {
                    if (on_path(q.A, q.B, bid)) {
                        results[i] = "no";
                    } else {
                        results[i] = "yes";
                    }
                } else {
                    // Not a bridge, removing edge doesn't disconnect
                    results[i] = "yes";
                }
            } else {
                // Should not happen given problem constraints
                results[i] = "yes"; 
            }
        } else {
            // Check if vertex C is on the path between A and B
            if (on_path(q.A, q.B, q.C)) {
                results[i] = "no";
            } else {
                results[i] = "yes";
            }
        }
    });
    
    for (int i = 0; i < Q; ++i) {
        cout << results[i] << "\n";
    }
    
    return 0;
}