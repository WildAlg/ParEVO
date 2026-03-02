/**
 * Problem: Baltazar's Magic Potion
 * 
 * Solution Approach:
 * 1. Calculate shortest path distances from the start node (1) and from the end node (n) using bidirectional Dijkstra.
 *    - We store 'visit_order' during Dijkstra to get nodes sorted by distance, avoiding extra sorting steps later.
 *    - We use modular arithmetic (hashing) with two moduli to count the number of shortest paths.
 * 2. Identify "bridges" on the Shortest Path DAG (SP-DAG).
 *    - An edge (u, v) is a bridge if it lies on a shortest path and carries all "flow" of shortest paths (waysS[u] * waysT[v] == total_ways).
 * 3. Bridges form a linear sequence along the SP-DAG. We index them 1 to K.
 * 4. We need to find bridges that can be bypassed by a path of length exactly D + 1.
 *    - For each node u, compute L[u]: the index of the last bridge on any shortest path from 1 to u.
 *    - For each node u, compute R[u]: the index of the first bridge on any shortest path from u to n.
 *    - L and R are computed by propagating values through the SP-DAG using the topological orders from Dijkstra.
 * 5. A bypass edge (u, v) with weight w such that distS[u] + w + distT[v] == D + 1 creates a path of length D+1.
 *    - This edge allows bypassing all bridges strictly between L[u] and R[v] (indices k where L[u] < k < R[v]).
 * 6. We use a difference array to count how many such bypasses cover each bridge.
 *    - Any bridge covered by at least one bypass of length D+1 is a valid solution.
 * 
 * Performance Optimizations:
 * - Fast I/O.
 * - Efficient Dijkstra implementation with subtraction-based modular arithmetic.
 * - Use of 'parlay' library for sorting bridges.
 * - Linear time propagation for L and R arrays.
 * 
 * Time Complexity: O(M log N)
 * Space Complexity: O(N + M)
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <tuple>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// Constants
const long long INF_LL = 1e18;
const int MOD1 = 1000000007;
const int MOD2 = 1000000009;

struct Edge {
    int to;
    int weight;
    int id;
};

struct FullEdge {
    int u, v, w, id;
};

struct Bridge {
    int u, v, id;
    long long dist;
};

// Dijkstra State
struct DState {
    long long d;
    int u;
    bool operator>(const DState& other) const { return d > other.d; }
};

void solve() {
    int n, m;
    if (!(cin >> n >> m)) return;

    // Adjacency list
    vector<vector<Edge>> adj(n + 1);
    vector<FullEdge> edges;
    edges.reserve(m);

    for (int i = 0; i < m; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w, i + 1});
        adj[v].push_back({u, w, i + 1});
        edges.push_back({u, v, w, i + 1});
    }

    // Dijkstra function that returns visit order (topological order for SP-DAG)
    auto run_dijkstra = [&](int start, vector<long long>& dist, vector<pair<int, int>>& ways, vector<int>& visit_order) {
        dist.assign(n + 1, INF_LL);
        ways.assign(n + 1, {0, 0});
        visit_order.clear();
        visit_order.reserve(n);
        
        dist[start] = 0;
        ways[start] = {1, 1};
        
        priority_queue<DState, vector<DState>, greater<DState>> pq;
        pq.push({0, start});

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            if (d > dist[u]) continue;
            visit_order.push_back(u);

            for (const auto& e : adj[u]) {
                long long new_dist = d + e.weight;
                if (new_dist < dist[e.to]) {
                    dist[e.to] = new_dist;
                    ways[e.to] = ways[u];
                    pq.push({new_dist, e.to});
                } else if (new_dist == dist[e.to]) {
                    // Fast modular addition
                    int w1 = ways[e.to].first + ways[u].first;
                    if (w1 >= MOD1) w1 -= MOD1;
                    ways[e.to].first = w1;

                    int w2 = ways[e.to].second + ways[u].second;
                    if (w2 >= MOD2) w2 -= MOD2;
                    ways[e.to].second = w2;
                }
            }
        }
    };

    vector<long long> distS, distT;
    vector<pair<int, int>> waysS, waysT;
    vector<int> visitS, visitT;

    // Run Dijkstra from S=1
    run_dijkstra(1, distS, waysS, visitS);
    
    long long D = distS[n];
    if (D == INF_LL) {
        cout << "0\n\n";
        return;
    }

    // Run Dijkstra from T=n
    run_dijkstra(n, distT, waysT, visitT);

    // Identify Bridges
    pair<int, int> total_ways = waysS[n];
    vector<Bridge> bridges;
    bridges.reserve(n); 
    
    for (const auto& e : edges) {
        bool on_sp_uv = (distS[e.u] + e.w + distT[e.v] == D);
        bool on_sp_vu = (distS[e.v] + e.w + distT[e.u] == D);
        
        if (on_sp_uv) {
            long long c1 = (1LL * waysS[e.u].first * waysT[e.v].first) % MOD1;
            long long c2 = (1LL * waysS[e.u].second * waysT[e.v].second) % MOD2;
            if (c1 == total_ways.first && c2 == total_ways.second) {
                bridges.push_back({e.u, e.v, e.id, distS[e.u]});
            }
        } else if (on_sp_vu) {
            long long c1 = (1LL * waysS[e.v].first * waysT[e.u].first) % MOD1;
            long long c2 = (1LL * waysS[e.v].second * waysT[e.u].second) % MOD2;
            if (c1 == total_ways.first && c2 == total_ways.second) {
                bridges.push_back({e.v, e.u, e.id, distS[e.v]});
            }
        }
    }

    // Sort bridges by distance from source using parlay
    parlay::sort_inplace(bridges, [](const Bridge& a, const Bridge& b) {
        return a.dist < b.dist;
    });

    int K = bridges.size();
    vector<int> edge_rank(m + 2, 0);
    vector<int> bridge_ids;
    bridge_ids.reserve(K);
    for (int i = 0; i < K; ++i) {
        edge_rank[bridges[i].id] = i + 1;
        bridge_ids.push_back(bridges[i].id);
    }

    // Compute L[u]: Last bridge index on shortest path S -> u
    // Process in topological order (visitS)
    vector<int> L(n + 1, -1);
    L[1] = 0;

    for (int u : visitS) {
        if (L[u] == -1) continue;
        for (const auto& e : adj[u]) {
            // Check if edge is on SP-DAG: distS[u] + w == distS[v]
            if (distS[u] + e.weight == distS[e.to]) {
                int val = L[u];
                if (edge_rank[e.id] != 0) val = edge_rank[e.id];
                
                // L[v] is min of incoming paths (common prefix)
                if (L[e.to] == -1 || val < L[e.to]) L[e.to] = val;
            }
        }
    }

    // Compute R[u]: First bridge index on shortest path u -> T
    // Process in topological order from T (visitT), which means processing nodes closer to T first
    vector<int> R(n + 1, -1);
    R[n] = K + 1;

    for (int u : visitT) {
        if (R[u] == -1) continue;
        for (const auto& e : adj[u]) {
            // Edge u->v in graph. Check if it's an incoming edge to u in SP-DAG(T)
            // i.e., distT[v] == distT[u] + w, meaning v is further from T than u.
            if (distT[e.to] == distT[u] + e.weight) {
                int val = R[u];
                if (edge_rank[e.id] != 0) val = edge_rank[e.id];
                
                // R[v] is max of incoming paths (common suffix)
                if (R[e.to] == -1 || val > R[e.to]) R[e.to] = val;
            }
        }
    }

    // Difference array to count valid bypasses
    vector<int> diff(K + 2, 0);
    for (const auto& e : edges) {
        auto check = [&](int u, int v) {
            // Check if edge forms a path of length D+1
            if (distS[u] != INF_LL && distT[v] != INF_LL && distS[u] + e.w + distT[v] == D + 1) {
                int l = L[u];
                int r = R[v];
                // Valid bypass range (l, r) corresponds to bridges with index k such that l < k < r
                if (l != -1 && r != -1 && l + 1 < r) {
                    diff[l + 1]++;
                    diff[r]--;
                }
            }
        };
        check(e.u, e.v);
        check(e.v, e.u);
    }

    // Collect results
    vector<int> ans;
    ans.reserve(K);
    int cur = 0;
    for (int i = 1; i <= K; ++i) {
        cur += diff[i];
        if (cur > 0) ans.push_back(bridge_ids[i - 1]);
    }

    sort(ans.begin(), ans.end());

    cout << ans.size() << "\n";
    for (int i = 0; i < ans.size(); ++i) {
        cout << ans[i] << (i == ans.size() - 1 ? "" : " ");
    }
    cout << "\n";
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int t;
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
}