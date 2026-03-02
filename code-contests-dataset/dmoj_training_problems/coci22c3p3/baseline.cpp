/**
 * Problem: Baltazar's Magic Potion
 * 
 * Solution Approach:
 * 1. Calculate shortest path distances from the start node (1) and from the end node (n) using Dijkstra's algorithm.
 *    We also count the number of shortest paths to identify bridges.
 * 2. Identify "bridges" on the Shortest Path DAG. These are edges that are part of ALL shortest paths from 1 to n.
 *    Only these edges are candidates because increasing the weight of a non-bridge edge on a shortest path
 *    would simply make another existing shortest path the new unique shortest path (or keep distance same),
 *    resulting in a distance increase of 0.
 * 3. For each bridge, we need to determine the length of the shortest path from 1 to n that does NOT use this bridge.
 *    Let this length be D'. If D' = D + 1, then increasing the bridge's weight by 2 (making the path through it D+2)
 *    will make the new shortest distance min(D+2, D') = D+1.
 * 4. To find D' efficiently for all bridges:
 *    - Compute L[u]: the index of the last bridge that is unavoidable on any shortest path from 1 to u.
 *    - Compute R[u]: the index of the first bridge that is unavoidable on any shortest path from u to n.
 *    - Any edge (u, v) with weight w provides a bypass path of length dist(1, u) + w + dist(v, n).
 *      This path bypasses all bridges strictly between L[u] and R[v].
 * 5. We collect all such bypass ranges and path lengths, sort them by length using parlay::sort, and use a 
 *    Disjoint Set Union (DSU) based sweep-line approach to find the minimum bypass length for each bridge.
 * 6. Finally, we check which bridges satisfy the condition min_bypass == D + 1.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <tuple>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

const long long INF_LL = 1e18;
const int MOD1 = 1000000007;
const int MOD2 = 1000000009;

struct Edge {
    int to;
    int weight;
    int id;
};

struct Update {
    int l, r;
    long long w;
    bool operator<(const Update& other) const {
        return w < other.w;
    }
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

    vector<vector<Edge>> adj(n + 1);
    vector<tuple<int, int, int, int>> all_edges; // u, v, w, id
    all_edges.reserve(m);

    for (int i = 0; i < m; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w, i + 1});
        adj[v].push_back({u, w, i + 1});
        all_edges.emplace_back(u, v, w, i + 1);
    }

    // Dijkstra's algorithm to find distances and count paths
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

            for (auto& e : adj[u]) {
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

    long long shortest_dist = distS[n];
    if (shortest_dist == INF_LL) {
        cout << "0\n\n";
        return;
    }

    // Identify bridges on the Shortest Path DAG
    vector<BridgeInfo> bridges;
    vector<int> edge_to_bridge_pos(m + 2, 0); 

    for (const auto& edge : all_edges) {
        int u, v, w, id;
        tie(u, v, w, id) = edge;

        // Check if edge is on shortest path and carries all flow
        // Direction u -> v
        if (distS[u] + w + distT[v] == shortest_dist) {
            long long wp1 = (waysS[u].first * waysT[v].first) % MOD1;
            long long wp2 = (waysS[u].second * waysT[v].second) % MOD2;
            if (wp1 == waysS[n].first && wp2 == waysS[n].second) {
                bridges.push_back({u, v, id, distS[u]});
            }
        }
        // Direction v -> u
        else if (distS[v] + w + distT[u] == shortest_dist) {
            long long wp1 = (waysS[v].first * waysT[u].first) % MOD1;
            long long wp2 = (waysS[v].second * waysT[u].second) % MOD2;
            if (wp1 == waysS[n].first && wp2 == waysS[n].second) {
                bridges.push_back({v, u, id, distS[v]});
            }
        }
    }

    // Sort bridges by distance from source to get their order on the path
    sort(bridges.begin(), bridges.end(), [](const BridgeInfo& a, const BridgeInfo& b) {
        return a.dist_u < b.dist_u;
    });

    int K = bridges.size();
    vector<int> bridge_ids;
    bridge_ids.reserve(K);
    for (int i = 0; i < K; ++i) {
        edge_to_bridge_pos[bridges[i].id] = i + 1;
        bridge_ids.push_back(bridges[i].id);
    }

    // Compute L[u]: Last necessary bridge index on path S -> u
    vector<int> nodes(n);
    for(int i=0; i<n; ++i) nodes[i] = i+1;
    sort(nodes.begin(), nodes.end(), [&](int a, int b){
        return distS[a] < distS[b];
    });

    vector<int> L(n + 1, 0);
    for (int u : nodes) {
        if (u == 1) continue;
        int current_L = -1;
        bool first = true;
        
        for (auto& e : adj[u]) {
            int p = e.to;
            int w = e.weight;
            // Check if p -> u is in SP DAG
            if (distS[p] + w == distS[u]) {
                int val = L[p];
                int id = e.id;
                if (edge_to_bridge_pos[id] != 0) {
                    val = edge_to_bridge_pos[id];
                }
                
                if (first) {
                    current_L = val;
                    first = false;
                } else {
                    current_L = min(current_L, val);
                }
            }
        }
        if (!first) L[u] = current_L;
    }

    // Compute R[u]: First necessary bridge index on path u -> T
    // Process in order of increasing distT (reverse topological for SP DAG from S)
    sort(nodes.begin(), nodes.end(), [&](int a, int b){
        return distT[a] < distT[b];
    });

    vector<int> R(n + 1, K + 1);
    for (int u : nodes) {
        if (u == n) continue;
        int current_R = -1;
        bool first = true;

        for (auto& e : adj[u]) {
            int v = e.to;
            int w = e.weight;
            // Check if u -> v is in SP DAG (equiv to v -> u in reversed graph)
            if (distT[v] + w == distT[u]) {
                int val = R[v];
                int id = e.id;
                if (edge_to_bridge_pos[id] != 0) {
                    val = edge_to_bridge_pos[id];
                }

                if (first) {
                    current_R = val;
                    first = false;
                } else {
                    current_R = max(current_R, val);
                }
            }
        }
        if (!first) R[u] = current_R;
    }

    // Collect bypass updates
    vector<Update> updates_vec;
    updates_vec.reserve(m);

    for (const auto& edge : all_edges) {
        int u, v, w, id;
        tie(u, v, w, id) = edge;

        if (edge_to_bridge_pos[id] != 0) continue; // Bridges cannot bypass themselves

        // Try path through u->v
        long long path_len = distS[u] + w + distT[v];
        if (path_len < INF_LL) { // Only valid paths
            int l = L[u];
            int r = R[v];
            if (l + 1 <= r - 1) {
                updates_vec.push_back({l + 1, r - 1, path_len});
            }
        }

        // Try path through v->u
        path_len = distS[v] + w + distT[u];
        if (path_len < INF_LL) {
            int l = L[v];
            int r = R[u];
            if (l + 1 <= r - 1) {
                updates_vec.push_back({l + 1, r - 1, path_len});
            }
        }
    }

    // Use parlay to sort updates efficiently
    parlay::sequence<Update> updates(updates_vec.begin(), updates_vec.end());
    auto sorted_updates = parlay::sort(updates);

    // Process updates using DSU to find min bypass for each bridge
    vector<long long> min_bypass(K + 2, INF_LL);
    vector<int> parent(K + 2);
    for(int i=0; i<=K+1; ++i) parent[i] = i;

    auto find_set = [&](auto&& self, int i) -> int {
        if (parent[i] == i) return i;
        return parent[i] = self(self, parent[i]);
    };

    for (const auto& upd : sorted_updates) {
        int l = upd.l;
        int r = upd.r;
        long long w = upd.w;
        
        for (int i = find_set(find_set, l); i <= r; i = find_set(find_set, i)) {
            min_bypass[i] = w;
            parent[i] = find_set(find_set, i + 1);
        }
    }

    // Collect result indices
    vector<int> result_indices;
    for (int i = 1; i <= K; ++i) {
        if (min_bypass[i] == shortest_dist + 1) {
            result_indices.push_back(bridge_ids[i - 1]);
        }
    }

    sort(result_indices.begin(), result_indices.end());

    cout << result_indices.size() << "\n";
    for (int i = 0; i < result_indices.size(); ++i) {
        cout << result_indices[i] << (i == result_indices.size() - 1 ? "" : " ");
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