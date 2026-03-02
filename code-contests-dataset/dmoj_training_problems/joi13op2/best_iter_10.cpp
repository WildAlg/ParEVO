/**
 * JOI Open Contest - Digital Lines / Synchronization
 * 
 * Solution Approach:
 * The problem asks for the number of different pieces of information contained in specific servers
 * after a sequence of edge modifications (building/removing lines).
 * 
 * We model the network as a rooted tree (rooted at 1).
 * Although edges are added/removed, the problem guarantees that if all edges were present, 
 * they form a tree. Thus, we can fix the parent-child relationships based on this spanning tree.
 * 
 * We maintain the state of each component. A component is a set of nodes connected by active edges.
 * For each component, we track the number of unique information pieces it contains.
 * This value is stored at the "root" of the component (the node closest to the global root 1).
 * 
 * We use a Binary Indexed Tree (BIT) to efficiently find the component root for any node.
 * The BIT maintains the number of *inactive* edges on the path from the global root to any node.
 * Two nodes u and v (where v is ancestor of u) are in the same component iff the number of 
 * inactive edges from root to u is the same as to v.
 * 
 * Operations:
 * 1. Build line (u, v): Activates edge. Components merge.
 *    The child node's component merges into the parent's component.
 *    New info count = val[parent_root] + val[child] - last_val[child].
 * 2. Remove line (u, v): Deactivates edge. Component splits.
 *    Child node becomes a new component root.
 *    It inherits the info count from the parent's component: val[child] = val[parent_root].
 *    We also update last_val[child] = val[parent_root] to track history.
 * 
 * Complexity: O(M log^2 N + Q log^2 N) time.
 * We use parlay::parallel_for for initialization and query processing.
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

// Global Variables
int N, M, Q;
vector<pair<int, int>> adj[MAXN]; // {neighbor, edge_index}
int parent_node[MAXN];
int up[MAXN][LOGN]; // Binary lifting table
int tin[MAXN], tout[MAXN]; // DFS entry/exit times
int timer;
int edge_to_node[MAXN]; // Maps edge index to the child node in the rooted tree
int val[MAXN]; // Number of pieces of info in component rooted at i
int last_val[MAXN]; // Value of component when i last disconnected
bool is_active[MAXN]; // Edge status (true = active/built, false = inactive)
int queries[MAXN];
int answers[MAXN];
int events[200005];

// Binary Indexed Tree for Range Update, Point Query
struct BIT {
    int tree[MAXN];

    // Add delta to index i
    void update(int i, int delta) {
        for (; i <= N; i += i & -i) tree[i] += delta;
    }

    // Add delta to range [l, r]
    void range_update(int l, int r, int delta) {
        update(l, delta);
        update(r + 1, -delta);
    }

    // Get value at index i
    int query(int i) {
        int sum = 0;
        for (; i > 0; i -= i & -i) sum += tree[i];
        return sum;
    }
} bit;

// DFS to build tree structure and LCA table
void dfs(int u, int p) {
    tin[u] = ++timer;
    parent_node[u] = p;
    up[u][0] = p;
    for (int i = 1; i < LOGN; i++) {
        up[u][i] = up[up[u][i-1]][i-1];
    }

    for (auto& edge : adj[u]) {
        int v = edge.first;
        int id = edge.second;
        if (v != p) {
            edge_to_node[id] = v;
            dfs(v, u);
        }
    }
    tout[u] = timer;
}

// Find the root of the active component containing u
// This is the highest ancestor 'a' of 'u' such that the path u->a has no inactive edges.
int find_comp_root(int u) {
    int target = bit.query(tin[u]);
    int curr = u;
    
    // Binary lifting to find highest ancestor with same BIT value (same number of inactive edges from root)
    for (int i = LOGN - 1; i >= 0; i--) {
        int anc = up[curr][i];
        if (anc != 0 && bit.query(tin[anc]) == target) {
            curr = anc;
        }
    }
    return curr;
}

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N >> M >> Q)) return 0;

    // Read edges
    for (int i = 1; i < N; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
    }

    // Read events
    for (int i = 1; i <= M; i++) {
        cin >> events[i];
    }

    // Read queries
    for (int i = 0; i < Q; i++) {
        cin >> queries[i];
    }

    // Initialize Tree Structure
    timer = 0;
    dfs(1, 0);

    // Initialize BIT: Initially all edges are inactive.
    // An inactive edge (u, parent[u]) contributes 1 to the count of inactive edges for u's subtree.
    // We start from 2 because root (1) has no parent edge.
    for (int i = 2; i <= N; i++) {
        bit.range_update(tin[i], tout[i], 1);
    }

    // Initialize component values in parallel
    parlay::parallel_for(1, N + 1, [&](int i) {
        val[i] = 1;       // Initially each server has 1 piece of info (itself)
        last_val[i] = 0;  // No history
        is_active[i] = false; // Note: is_active uses edge indices 1..N-1
    });

    // Process events sequentially
    for (int t = 1; t <= M; t++) {
        int e_idx = events[t];
        int u = edge_to_node[e_idx]; // The child node corresponding to the edge
        
        if (is_active[e_idx]) {
            // Remove line: Edge becomes inactive
            // 1. Mark edge as inactive in BIT (add 1 to subtree)
            bit.range_update(tin[u], tout[u], 1);
            is_active[e_idx] = false;

            // 2. Component Split
            // u becomes the root of a new component.
            // Before split, u was part of parent[u]'s component.
            // We need to copy the full info from that component to u.
            int p = parent_node[u];
            int root_p = find_comp_root(p);
            
            val[u] = val[root_p];
            last_val[u] = val[root_p];
            
        } else {
            // Build line: Edge becomes active
            // 1. Mark edge as active in BIT (subtract 1 from subtree)
            bit.range_update(tin[u], tout[u], -1);
            is_active[e_idx] = true;

            // 2. Component Merge
            // u's component merges into parent[u]'s component.
            // The new info u brings is (val[u] - last_val[u]).
            int p = parent_node[u];
            int root_p = find_comp_root(p);
            
            val[root_p] += val[u] - last_val[u];
        }
    }

    // Answer queries in parallel
    // At time M+1, the structure is static.
    parlay::parallel_for(0, Q, [&](int i) {
        int u = queries[i];
        int root = find_comp_root(u);
        answers[i] = val[root];
    });

    for (int i = 0; i < Q; i++) {
        cout << answers[i] << "\n";
    }

    return 0;
}