#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

const long long INF = 1e16;

int N, M, Q;
vector<long long> D;
vector<vector<int>> adj;
vector<int> sz;
vector<bool> visited;

// dp[u][k][state]
// k: number of people exchanging solutions in the subtree
// State 0: u is NOT relaxed.
// State 1: u is relaxed but ISOLATED (needs a parent to connect).
// State 2: u is relaxed and SATISFIED (has a relaxed child).
long long dp[1005][1005][3];

void dfs(int u, int p) {
    sz[u] = 1;
    
    // Initialize leaf states
    // A single node contributes 0 to the count 'k'.
    dp[u][0][0] = 0;        // Not relaxed
    dp[u][0][1] = D[u];     // Relaxed, waiting for connection
    dp[u][0][2] = INF;      // Impossible to be satisfied alone
    
    // Initialize larger k to INF
    for (int i = 1; i <= N; i++) {
        dp[u][i][0] = dp[u][i][1] = dp[u][i][2] = INF;
    }

    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);

        // Temporary DP array for merging u and child v
        static long long next_dp[1005][3];
        int new_sz = sz[u] + sz[v];

        // Initialize temporary array
        for (int i = 0; i <= new_sz; i++) {
            next_dp[i][0] = next_dp[i][1] = next_dp[i][2] = INF;
        }

        // Convolve current u states with child v states
        for (int i = 0; i <= sz[u]; i++) {
            // Optimization: Skip invalid starting states
            if (dp[u][i][0] == INF && dp[u][i][1] == INF && dp[u][i][2] == INF) continue;

            for (int j = 0; j <= sz[v]; j++) {
                if (dp[v][j][0] == INF && dp[v][j][1] == INF && dp[v][j][2] == INF) continue;

                long long u0 = dp[u][i][0];
                long long u1 = dp[u][i][1];
                long long u2 = dp[u][i][2];
                
                long long v0 = dp[v][j][0];
                long long v1 = dp[v][j][1]; // Child relaxed, isolated
                long long v2 = dp[v][j][2]; // Child relaxed, satisfied

                // 1. Target: u is State 0 (Not Relaxed)
                // u must be 0. v can be 0 or 2. v cannot be 1 (it would stay isolated forever).
                if (u0 != INF) {
                    if (v0 != INF) next_dp[i + j][0] = min(next_dp[i + j][0], u0 + v0);
                    if (v2 != INF) next_dp[i + j][0] = min(next_dp[i + j][0], u0 + v2);
                }

                // 2. Target: u is State 1 (Relaxed, Isolated)
                // u must be 1. v can be 0 or 2. v cannot be 1 (connection would satisfy u).
                if (u1 != INF) {
                    if (v0 != INF) next_dp[i + j][1] = min(next_dp[i + j][1], u1 + v0);
                    if (v2 != INF) next_dp[i + j][1] = min(next_dp[i + j][1], u1 + v2);
                }

                // 3. Target: u is State 2 (Relaxed, Satisfied)
                
                // Case A: u was already satisfied (State 2)
                if (u2 != INF) {
                    if (v0 != INF) next_dp[i + j][2] = min(next_dp[i + j][2], u2 + v0);
                    // If v is 1, it connects to u. v becomes satisfied (+1 count). u stays satisfied.
                    if (v1 != INF) next_dp[i + j + 1][2] = min(next_dp[i + j + 1][2], u2 + v1);
                    if (v2 != INF) next_dp[i + j][2] = min(next_dp[i + j][2], u2 + v2);
                }

                // Case B: u was isolated (State 1), but v satisfies u
                if (u1 != INF) {
                    // u(1) + v(1) -> Both connect and become satisfied. Count increases by +2.
                    if (v1 != INF) next_dp[i + j + 2][2] = min(next_dp[i + j + 2][2], u1 + v1);
                    // u(1) + v(2) -> u sees v is relaxed. u becomes satisfied. v already was. Count +1.
                    if (v2 != INF) next_dp[i + j + 1][2] = min(next_dp[i + j + 1][2], u1 + v2);
                }
            }
        }
        
        // Update u with merged results
        sz[u] = new_sz;
        for (int i = 0; i <= sz[u]; i++) {
            dp[u][i][0] = next_dp[i][0];
            dp[u][i][1] = next_dp[i][1];
            dp[u][i][2] = next_dp[i][2];
        }
    }
}

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N >> M)) return 0;

    D.resize(N + 1);
    for (int i = 1; i <= N; i++) cin >> D[i];

    adj.resize(N + 1);
    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    sz.resize(N + 1);
    visited.resize(N + 1, false);

    // Global DP: min cost to get exactly k people exchanging solutions across the entire forest
    vector<long long> global_dp(1, 0); // Base case: 0 cost for 0 people

    // Iterate over all components (Forest)
    for (int i = 1; i <= N; i++) {
        if (!visited[i]) {
            // BFS to mark component as visited
            vector<int> q;
            q.push_back(i);
            visited[i] = true;
            int head = 0;
            while(head < q.size()){
                int u = q[head++];
                for(int v : adj[u]){
                    if(!visited[v]){
                        visited[v] = true;
                        q.push_back(v);
                    }
                }
            }
            
            // Run Tree DP on this component
            dfs(i, 0);

            // Extract valid results for this component
            // For the root, State 1 is invalid (isolated root implies wasted cost),
            // so we only take min(State 0, State 2).
            vector<long long> tree_res(sz[i] + 1, INF);
            for(int k = 0; k <= sz[i]; ++k) {
                tree_res[k] = min(dp[i][k][0], dp[i][k][2]);
            }

            // Merge component results into global results (Knapsack)
            vector<long long> next_global(global_dp.size() + sz[i], INF);
            for (int k = 0; k < global_dp.size(); k++) {
                if (global_dp[k] == INF) continue;
                for (int l = 0; l <= sz[i]; l++) {
                    if (tree_res[l] == INF) continue;
                    next_global[k + l] = min(next_global[k + l], global_dp[k] + tree_res[l]);
                }
            }
            global_dp = next_global;
        }
    }

    // Precompute answers for queries
    // Store pairs of (Cost, Max_People)
    vector<pair<long long, int>> lookup;
    for (int k = 0; k < global_dp.size(); k++) {
        if (global_dp[k] < INF) {
            lookup.push_back({global_dp[k], k});
        }
    }
    sort(lookup.begin(), lookup.end());

    // Make lookup monotonic: for a given cost, we want the max possible k
    vector<pair<long long, int>> final_lookup;
    int max_k = -1;
    for (auto p : lookup) {
        if (p.second > max_k) {
            max_k = p.second;
            final_lookup.push_back({p.first, max_k});
        }
    }

    cin >> Q;
    for (int i = 0; i < Q; i++) {
        long long S;
        cin >> S;
        
        // Find the largest cost <= S
        // upper_bound returns the first element > S, so we step back
        auto it = upper_bound(final_lookup.begin(), final_lookup.end(), make_pair(S, N + 1));
        
        if (it == final_lookup.begin()) {
            cout << "0\n";
        } else {
            cout << prev(it)->second << "\n";
        }
    }

    return 0;
}