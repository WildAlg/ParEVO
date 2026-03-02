/**
 * Problem: Baltazar's Magic Potion
 * 
 * Solution Approach:
 * 1. Compute shortest path distances from the start node (1) and from the end node (n) using Dijkstra's algorithm.
 *    We also count the number of shortest paths modulo two large primes to robustly identify bridges.
 * 2. Identify "bridges" on the Shortest Path DAG. These are edges that are part of ALL shortest paths from 1 to n.
 *    Only bridges are candidates because increasing a non-bridge edge on a shortest path wouldn't force the
 *    shortest path length to increase (an alternative path of same length D exists).
 * 3. For the shortest distance to increase by exactly 1 (to D+1), there must be an alternative path of length D+1
 *    that bypasses the bridge. If the shortest bypass is > D+1, the new distance would be D+2 (via the bridge).
 * 4. We compute:
 *    - L[u]: The index of the last bridge on the path from 1 to u.
 *    - R[u]: The index of the first bridge on the path from u to n.
 * 5. Any edge (u, v) with weight w such that distS[u] + w + distT[v] == D + 1 is a "bypass" edge.
 *    This edge allows bypassing all bridges strictly between L[u] and R[v].
 * 6. We use a difference array to count how many such bypasses cover each bridge.
 *    Any bridge covered by at least one bypass of length D+1 is a valid solution.
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
const long long MOD1 = 1000000007;
const long long MOD2 = 1000000009;

struct Edge {
    int to;
    int weight;
    int id;
};

struct BridgeInfo {
    int u, v, id;
    long long dist_u;
};

struct State {
    long long dist;
    int u;
    bool operator>(const State& other) const {
        return dist > other.dist;
    }
};

void solve() {
    int n, m;
    if (!(cin >> n >> m)) return;

    // Adjacency list (1-based)
    vector<vector<Edge>> adj(n + 1);
    // Store raw edges for iteration
    struct RawEdge { int u, v, w, id; };
    vector<RawEdge> all_edges;
    all_edges.reserve(m);

    for (int i = 0; i < m; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w, i + 1});
        adj[v].push_back({u, w, i + 1});
        all_edges.push_back({u, v, w, i + 1});
    }

    // Dijkstra's Algorithm
    auto run_dijkstra = [&](int start, vector<long long>& dist, vector<pair<long long, long long>>& ways) {
        dist.assign(n + 1, INF_LL);
        ways.assign(n + 1, {0, 0});
        
        dist[start] = 0;
        ways[start] = {1, 1};
        
        priority_queue<State, vector<State>, greater<State>> pq;
        pq.push({0, start});

        while (!pq.empty()) {
            long long d = pq.top().dist;
            int u = pq.top().u;
            pq.pop();

            if (d > dist[u]) continue;

            for (const auto& e : adj[u]) {
                if (dist[u] + e.weight < dist[e.to]) {
                    dist[e.to] = dist[u] + e.weight;
                    ways[e.to] = ways[u];
                    pq.push({dist[e.to], e.to});
                } else if (dist[u] + e.weight == dist[e.to]) {
                    ways[e.to].first = (ways[e.to].first + ways[u].first) % MOD1;
                    ways[e.to].second = (ways[e.to].second + ways[u].second) % MOD2;
                }
            }
        }
    };

    vector<long long> distS, distT;
    vector<pair<long long, long long>> waysS, waysT;

    run_dijkstra(1, distS, waysS);
    run_dijkstra(n, distT, waysT);

    long long D = distS[n];
    if (D == INF_LL) {
        cout << "0\n\n";
        return;
    }

    // Identify Bridges
    vector<BridgeInfo> bridges;
    long long tot1 = waysS[n].first;
    long long tot2 = waysS[n].second;

    for (const auto& e : all_edges) {
        // Check u -> v
        if (distS[e.u] + e.w + distT[e.v] == D) {
            if ((waysS[e.u].first * waysT[e.v].first) % MOD1 == tot1 &&
                (waysS[e.u].second * waysT[e.v].second) % MOD2 == tot2) {
                bridges.push_back({e.u, e.v, e.id, distS[e.u]});
            }
        } 
        // Check v -> u
        else if (distS[e.v] + e.w + distT[e.u] == D) {
            if ((waysS[e.v].first * waysT[e.u].first) % MOD1 == tot1 &&
                (waysS[e.v].second * waysT[e.u].second) % MOD2 == tot2) {
                bridges.push_back({e.v, e.u, e.id, distS[e.v]});
            }
        }
    }

    // Sort bridges by distance from source
    parlay::sort_inplace(bridges, [](const BridgeInfo& a, const BridgeInfo& b) {
        return a.dist_u < b.dist_u;
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
    parlay::sequence<int> nodes(n);
    for(int i=0; i<n; ++i) nodes[i] = i+1;
    
    // Process in topological order (increasing distS)
    parlay::sort_inplace(nodes, [&](int a, int b) { return distS[a] < distS[b]; });
    
    vector<int> L(n + 1, -1);
    L[1] = 0;
    
    for (int u : nodes) {
        if (L[u] == -1) continue;
        for (const auto& e : adj[u]) {
            if (distS[u] + e.weight == distS[e.to]) {
                int val = (edge_rank[e.id] != 0) ? edge_rank[e.id] : L[u];
                // L[v] should be the minimum of all incoming paths (common prefix of bridges)
                if (L[e.to] == -1 || val < L[e.to]) L[e.to] = val;
            }
        }
    }

    // Compute R[u]: First bridge index on shortest path u -> T
    // Process in reverse topological order (increasing distT)
    parlay::sort_inplace(nodes, [&](int a, int b) { return distT[a] < distT[b]; });
    
    vector<int> R(n + 1, -1);
    R[n] = K + 1;
    
    for (int u : nodes) {
        if (R[u] == -1) continue;
        for (const auto& e : adj[u]) {
            // Edge u->v in graph corresponds to v->u in SP-DAG towards T
            // Condition: distT[v] == distT[u] + w
            if (distT[e.to] == distT[u] + e.weight) {
                int val = (edge_rank[e.id] != 0) ? edge_rank[e.id] : R[u];
                // R[v] should be the maximum (common suffix of bridges)
                if (R[e.to] == -1 || val > R[e.to]) R[e.to] = val;
            }
        }
    }

    // Difference array to count valid bypasses
    vector<int> diff(K + 2, 0);
    
    for (const auto& e : all_edges) {
        auto check = [&](int u, int v) {
            // Check if edge forms a path of length D+1
            if (distS[u] != INF_LL && distT[v] != INF_LL && distS[u] + e.w + distT[v] == D + 1) {
                int l = L[u];
                int r = R[v];
                // Bypasses bridges strictly between l and r
                if (l != -1 && r != -1 && l + 1 <= r - 1) {
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