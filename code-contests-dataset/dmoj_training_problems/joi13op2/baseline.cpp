/**
 * JOI Open Contest - Digital Lines / Synchronization
 * 
 * Solution Approach:
 * The problem asks for the number of different pieces of information contained in specific servers
 * after a sequence of edge modifications (building/removing lines).
 * This is equivalent to the "JOI Synchronization" problem.
 * 
 * Key Mechanics:
 * - When a line connects two components, they synchronize, meaning both components now possess
 *   the union of information pieces from both original components.
 * - When a line is removed, the components split, but retain all information they had at that moment.
 * 
 * Algorithm:
 * 1. We model the network as a rooted tree (arbitrarily rooted at 1).
 * 2. We maintain the "active" components. In the rooted tree, an active component corresponds to
 *    a connected subgraph of active edges. Each component is identified by its root (the node
 *    closest to the global root 1).
 * 3. We track the amount of information `val[r]` for each component root `r`.
 *    - Initially, every node is its own component with `val[i] = 1`.
 * 4. We also track `last_val[u]`, which stores the amount of information `u` had just before it
 *    was last disconnected from its parent. This helps in calculating the "new" information `u` brings
 *    when reconnected.
 * 5. Operations:
 *    - **Build Line (u, parent[u])**: The component rooted at `u` merges into the component of `parent[u]`.
 *      Let `R` be the root of `parent[u]`'s component.
 *      The new information count is `val[R] += val[u] - last_val[u]`.
 *    - **Remove Line (u, parent[u])**: The component splits. `u` becomes a new root.
 *      Just before split, `u` was part of `R`'s component. So `u` inherits `val[R]`.
 *      `val[u] = val[R]`.
 *      `last_val[u] = val[R]` (to record state at disconnection).
 * 
 * Implementation Details:
 * - To efficiently find the component root `R` for any node, we use a Binary Indexed Tree (BIT)
 *   combined with Binary Lifting.
 * - The BIT maintains the number of inactive edges on the path from the global root to any node `v`.
 *   This allows us to check if two nodes are in the same component (same number of inactive edges above them).
 * - Binary Lifting allows us to jump up the tree to find the highest ancestor in the same component in O(log^2 N).
 * - Parlay library is used for parallel initialization and parallel query processing.
 * 
 * Complexity: O((N + M) log^2 N) time, O(N log N) space.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

using namespace std;

// Constants
const int MAXN = 100005;
const int LOGN = 18; // Sufficient for N <= 100,000

// Globals
int N, M, Q;
vector<pair<int, int>> adj[MAXN]; // {neighbor, edge_index}
pair<int, int> edge_list[MAXN]; 
int edge_to_child[MAXN]; // Maps edge index to the child node in the rooted tree
int parent[MAXN];
int up[MAXN][LOGN]; // Binary lifting table
int tin[MAXN], tout[MAXN]; // DFS entry/exit times
int timer;
int val[MAXN]; // Current information count for component roots
int last_val[MAXN]; // Information count at last disconnect
bool edge_active[MAXN]; // State of each edge
int D[200005]; // Events
int queries[MAXN];
int ans[MAXN];

// Binary Indexed Tree for Range Update, Point Query
// Maintains 'dist(u)': number of inactive edges from root to u.
struct BIT {
    int tree[MAXN];
    
    // Add delta to index idx
    void update(int idx, int delta) {
        for (; idx <= N; idx += idx & -idx)
            tree[idx] += delta;
    }
    
    // Add delta to range [l, r]
    void range_update(int l, int r, int delta) {
        update(l, delta);
        update(r + 1, -delta);
    }
    
    // Get value at index idx
    int query(int idx) {
        int sum = 0;
        for (; idx > 0; idx -= idx & -idx)
            sum += tree[idx];
        return sum;
    }
} bit;

// DFS for tree setup (LCA table, parent pointers, DFS order)
void dfs(int u, int p) {
    tin[u] = ++timer;
    parent[u] = p;
    up[u][0] = p;
    for (int i = 1; i < LOGN; i++) {
        up[u][i] = up[up[u][i-1]][i-1];
    }
    
    for (auto& edge : adj[u]) {
        int v = edge.first;
        int idx = edge.second;
        if (v != p) {
            edge_to_child[idx] = v;
            dfs(v, u);
        }
    }
    tout[u] = timer;
}

// Find the root of the active component containing u
// This is the highest ancestor 'a' of 'u' such that the path u->a has no inactive edges.
// In terms of BIT values: find highest 'a' such that bit.query(tin[a]) == bit.query(tin[u]).
int find_comp_root(int u) {
    int target = bit.query(tin[u]);
    int curr = u;
    
    // Try to jump up as high as possible while staying in the same component
    for (int i = LOGN - 1; i >= 0; i--) {
        int ancestor = up[curr][i];
        // Check if ancestor is valid (non-zero) and in same component
        if (ancestor != 0 && bit.query(tin[ancestor]) == target) {
            curr = ancestor;
        }
    }
    return curr;
}

int main() {
    // Optimize I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    if (!(cin >> N >> M >> Q)) return 0;
    
    // Read edges
    for (int i = 1; i < N; i++) {
        cin >> edge_list[i].first >> edge_list[i].second;
        adj[edge_list[i].first].push_back({edge_list[i].second, i});
        adj[edge_list[i].second].push_back({edge_list[i].first, i});
    }
    
    // Read events
    for (int i = 1; i <= M; i++) {
        cin >> D[i];
    }
    
    // Read queries
    for (int i = 0; i < Q; i++) {
        cin >> queries[i];
    }
    
    // Initialize tree structure
    timer = 0;
    dfs(1, 0);
    
    // Initially all edges are inactive.
    // Each edge (u, parent[u]) contributes 1 to the inactive count of u's subtree.
    for (int i = 2; i <= N; i++) {
        bit.range_update(tin[i], tout[i], 1);
    }
    
    // Initialize component values in parallel
    parlay::parallel_for(1, N + 1, [&](int i) {
        val[i] = 1;
        last_val[i] = 0;
        edge_active[i] = false; 
    });
    
    // Process events sequentially
    for (int t = 1; t <= M; t++) {
        int e_idx = D[t];
        int u = edge_to_child[e_idx]; // The child node corresponding to edge
        
        if (!edge_active[e_idx]) {
            // Build line: Edge (u, parent[u]) becomes active.
            // Inactive count decreases by 1 for subtree of u.
            bit.range_update(tin[u], tout[u], -1);
            edge_active[e_idx] = true;
            
            // Merge components
            // u is now connected to parent[u].
            // Find root of parent's component
            int p = parent[u];
            int root_p = find_comp_root(p);
            
            // Update the root's value with the new information from u's component
            val[root_p] += val[u] - last_val[u];
            
        } else {
            // Remove line: Edge (u, parent[u]) becomes inactive.
            // Inactive count increases by 1 for subtree of u.
            bit.range_update(tin[u], tout[u], 1);
            edge_active[e_idx] = false;
            
            // Split components
            // u becomes a new root.
            // It inherits the full information of the component it was just part of.
            int p = parent[u];
            int root_p = find_comp_root(p); // This finds the root of p (which is still the old root)
            
            val[u] = val[root_p];
            last_val[u] = val[root_p];
        }
    }
    
    // Answer queries in parallel
    parlay::parallel_for(0, Q, [&](int i) {
        int u = queries[i];
        int root = find_comp_root(u);
        ans[i] = val[root];
    });
    
    for (int i = 0; i < Q; i++) {
        cout << ans[i] << "\n";
    }
    
    return 0;
}