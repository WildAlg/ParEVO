/*
 * Solution for IOI '08 - Cairo, Egypt
 * 
 * Algorithm Overview:
 * 1. The problem graph is a collection of functional components (each node has exactly one outgoing edge).
 *    This means each component consists of exactly one cycle with several trees rooted on the cycle nodes,
 *    directed towards the cycle.
 * 2. We can move between components using ferries. Since we want to maximize the total walking distance,
 *    and we can visit components in sequence, the answer is the sum of the maximum walking distances
 *    within each connected component.
 * 3. For each component, the maximum path can be either:
 *    a) A path entirely within one of the trees attached to a cycle node.
 *    b) A path that starts in a tree, goes to the cycle, traverses part of the cycle, and ends in another tree.
 * 4. We use a topological sort (peeling) to process the trees. This computes the maximum depth of each tree
 *    rooted at a cycle node and the maximum diameter within each tree.
 * 5. After peeling, we are left with disjoint cycles. For each cycle, we find the longest path that uses
 *    cycle edges. This is equivalent to finding max(H[u] + H[v] + dist(u, v)) for nodes u, v on the cycle,
 *    where H[x] is the depth of the tree attached to x. This is solved using a sliding window maximum
 *    on a linearized (doubled) version of the cycle.
 * 6. The parlay library is used for parallel initialization of data structures.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <deque>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Maximum number of islands as per constraints
const int MAXN = 1000005;

// Graph storage
// adj_to[i] stores the target of the bridge from island i
// adj_w[i] stores the length of the bridge from island i
int adj_to[MAXN];
int adj_w[MAXN];
int deg[MAXN]; // In-degree for topological sort

// Processing arrays
long long depth_val[MAXN]; // Stores max depth of tree attached to node i (from peeling)
long long comp_ans[MAXN];  // Stores the max path found so far for the component
int dsu_parent[MAXN];      // DSU parent for component identification
bool visited_cycle[MAXN];  // To mark cycles as processed
bool processed[MAXN];      // To mark nodes processed by topological sort

// DSU find with path compression (iterative to avoid recursion depth issues)
int find_set(int v) {
    int root = v;
    while (root != dsu_parent[root]) {
        root = dsu_parent[root];
    }
    int curr = v;
    while (curr != root) {
        int next = dsu_parent[curr];
        dsu_parent[curr] = root;
        curr = next;
    }
    return root;
}

// DSU union
void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) dsu_parent[b] = a;
}

int main() {
    // Optimize standard I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N;
    if (!(cin >> N)) return 0;

    // Initialize arrays in parallel using parlay
    parlay::parallel_for(0, N + 1, [&](int i) {
        dsu_parent[i] = i;
        deg[i] = 0;
        depth_val[i] = 0;
        comp_ans[i] = 0;
        visited_cycle[i] = false;
        processed[i] = false;
    });

    // Read input
    for (int i = 1; i <= N; ++i) {
        cin >> adj_to[i] >> adj_w[i];
    }

    // Build components using DSU and compute in-degrees
    for (int i = 1; i <= N; ++i) {
        deg[adj_to[i]]++;
        union_sets(i, adj_to[i]);
    }

    // Topological sort (peeling trees)
    // Nodes with in-degree 0 are leaves of the trees directed towards the cycle.
    vector<int> q;
    q.reserve(N);
    for (int i = 1; i <= N; ++i) {
        if (deg[i] == 0) {
            q.push_back(i);
        }
    }

    int head = 0;
    while(head < q.size()){
        int u = q[head++];
        processed[u] = true;
        int v = adj_to[u];
        int w = adj_w[u];
        
        int root = find_set(v);
        // Update diameter in the tree component.
        // The longest path passing through v using branch u is depth_val[v] + (depth_val[u] + w).
        // depth_val[v] currently holds the max depth from previously processed children.
        comp_ans[root] = max(comp_ans[root], depth_val[v] + depth_val[u] + w);
        
        // Update max depth for the parent node v
        depth_val[v] = max(depth_val[v], depth_val[u] + w);
        
        deg[v]--;
        if (deg[v] == 0) {
            q.push_back(v);
        }
    }

    // Process cycles
    // Any node not processed by topological sort is part of a cycle.
    for (int i = 1; i <= N; ++i) {
        if (!processed[i] && !visited_cycle[i]) {
            int root = find_set(i);
            
            // Extract cycle nodes
            vector<int> cycle_nodes;
            int curr = i;
            while (!visited_cycle[curr]) {
                visited_cycle[curr] = true;
                cycle_nodes.push_back(curr);
                curr = adj_to[curr];
            }
            
            int k = cycle_nodes.size();
            
            // Linearize cycle (doubled to handle wrap-around)
            // H[j] stores the max tree depth at the j-th cycle node
            // P[j] stores the prefix sum of edge weights along the cycle
            vector<long long> H(2 * k);
            vector<long long> P(2 * k + 1, 0);
            
            for (int j = 0; j < k; ++j) {
                int u = cycle_nodes[j];
                H[j] = depth_val[u];
                H[j + k] = depth_val[u];
                // Edge weight from u to next node in cycle
                P[j + 1] = adj_w[u];
                P[j + k + 1] = adj_w[u];
            }
            
            // Compute prefix sums
            for (int j = 1; j <= 2 * k; ++j) {
                P[j] += P[j-1];
            }
            
            // Sliding window maximum to find max(H[x] - P[x] + H[y] + P[y]) for x < y < x + k
            // This corresponds to the longest path between two nodes on the cycle plus their tree depths.
            deque<int> dq;
            long long max_cycle_path = 0;
            
            for (int y = 0; y < 2 * k; ++y) {
                // Remove indices that are out of the window of length k
                // We consider start points x in range [y - k + 1, y - 1]
                while (!dq.empty() && dq.front() < y - k + 1) {
                    dq.pop_front();
                }
                
                if (!dq.empty()) {
                    int x = dq.front();
                    max_cycle_path = max(max_cycle_path, (H[x] - P[x]) + (H[y] + P[y]));
                }
                
                // Maintain deque for max (H[y] - P[y]) in decreasing order
                while (!dq.empty() && (H[dq.back()] - P[dq.back()]) <= (H[y] - P[y])) {
                    dq.pop_back();
                }
                dq.push_back(y);
            }
            
            comp_ans[root] = max(comp_ans[root], max_cycle_path);
        }
    }

    // Sum up answers from all components
    long long total_ans = 0;
    for (int i = 1; i <= N; ++i) {
        if (dsu_parent[i] == i) {
            total_ans += comp_ans[i];
        }
    }

    cout << total_ans << endl;

    return 0;
}