//EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

using namespace std;

const long long INF = 1e18;

struct Node {
    int id;
    long long d;
    vector<int> adj;
};

// dp[state][k] = min cost
// state 0: u is not relaxed
// state 1: u is relaxed, but has no relaxed children (isolated in subtree context)
// state 2: u is relaxed, and has at least one relaxed child (connected in subtree context)
using DPState = vector<vector<long long>>;

vector<Node> nodes;
vector<bool> visited;

// Merge child results into parent results
void merge(DPState& parent_dp, const DPState& child_dp, int p_sz, int c_sz) {
    int new_sz = p_sz + c_sz;
    DPState next_dp(3, vector<long long>(new_sz + 1, INF));

    // Parent State 0
    for (int i = 0; i <= p_sz; ++i) {
        if (parent_dp[0][i] == INF) continue;
        for (int j = 0; j <= c_sz; ++j) {
            long long cost_v = child_dp[0][j];
            if (child_dp[1][j] < cost_v) cost_v = child_dp[1][j];
            if (child_dp[2][j] < cost_v) cost_v = child_dp[2][j];
            
            if (cost_v != INF) {
                next_dp[0][i + j] = min(next_dp[0][i + j], parent_dp[0][i] + cost_v);
            }
        }
    }

    // Parent State 1
    for (int i = 0; i <= p_sz; ++i) {
        if (parent_dp[1][i] == INF) continue;
        for (int j = 0; j <= c_sz; ++j) {
            if (child_dp[0][j] != INF) {
                next_dp[1][i + j] = min(next_dp[1][i + j], parent_dp[1][i] + child_dp[0][j]);
            }
        }
    }

    // Parent State 2
    for (int i = 0; i <= p_sz; ++i) {
        if (parent_dp[2][i] != INF) {
            for (int j = 0; j <= c_sz; ++j) {
                if (child_dp[0][j] != INF) next_dp[2][i + j] = min(next_dp[2][i + j], parent_dp[2][i] + child_dp[0][j]);
                if (child_dp[1][j] != INF) next_dp[2][i + j + 1] = min(next_dp[2][i + j + 1], parent_dp[2][i] + child_dp[1][j]);
                if (child_dp[2][j] != INF) next_dp[2][i + j] = min(next_dp[2][i + j], parent_dp[2][i] + child_dp[2][j]);
            }
        }
        
        if (parent_dp[1][i] != INF) {
            for (int j = 0; j <= c_sz; ++j) {
                if (child_dp[1][j] != INF) next_dp[2][i + j + 2] = min(next_dp[2][i + j + 2], parent_dp[1][i] + child_dp[1][j]);
                if (child_dp[2][j] != INF) next_dp[2][i + j + 1] = min(next_dp[2][i + j + 1], parent_dp[1][i] + child_dp[2][j]);
            }
        }
    }
    
    parent_dp = move(next_dp);
}

pair<DPState, int> solve_tree(int u, int p) {
    int sz = 1;
    DPState dp(3, vector<long long>(2, INF));
    
    dp[0][0] = 0;
    dp[1][0] = nodes[u].d;
    
    for (int v : nodes[u].adj) {
        if (v == p) continue;
        pair<DPState, int> res = solve_tree(v, u);
        merge(dp, res.first, sz, res.second);
        sz += res.second;
    }
    return {dp, sz};
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int N, M;
    if (!(cin >> N >> M)) return 0;
    
    nodes.resize(N + 1);
    for (int i = 1; i <= N; ++i) {
        cin >> nodes[i].d;
        nodes[i].id = i;
    }
    
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        nodes[u].adj.push_back(v);
        nodes[v].adj.push_back(u);
    }
    
    visited.assign(N + 1, false);
    vector<int> roots;
    
    for (int i = 1; i <= N; ++i) {
        if (!visited[i]) {
            roots.push_back(i);
            vector<int> q = {i};
            visited[i] = true;
            int head = 0;
            while(head < q.size()){
                int u = q[head++];
                for(int v : nodes[u].adj){
                    if(!visited[v]){
                        visited[v] = true;
                        q.push_back(v);
                    }
                }
            }
        }
    }
    
    parlay::sequence<int> root_seq(roots.begin(), roots.end());
    
    auto tree_results = parlay::map(root_seq, [&](int root) {
        pair<DPState, int> res = solve_tree(root, 0);
        vector<long long> costs(res.second + 1);
        for (int k = 0; k <= res.second; ++k) {
            costs[k] = res.first[0][k];
            if (res.first[2][k] < costs[k]) costs[k] = res.first[2][k];
        }
        return costs;
    });
    
    vector<long long> global_dp = {0};
    
    for (const auto& tree_costs : tree_results) {
        int current_sz = global_dp.size() - 1;
        int tree_sz = tree_costs.size() - 1;
        vector<long long> next_dp(current_sz + tree_sz + 1, INF);
        
        for (int i = 0; i <= current_sz; ++i) {
            if (global_dp[i] == INF) continue;
            for (int j = 0; j <= tree_sz; ++j) {
                if (tree_costs[j] != INF) {
                    next_dp[i + j] = min(next_dp[i + j], global_dp[i] + tree_costs[j]);
                }
            }
        }
        global_dp = move(next_dp);
    }
    
    vector<pair<long long, int>> sorted_costs;
    for (int k = 0; k < global_dp.size(); ++k) {
        if (global_dp[k] != INF) {
            sorted_costs.push_back({global_dp[k], k});
        }
    }
    sort(sorted_costs.begin(), sorted_costs.end());
    
    vector<pair<long long, int>> query_lookup;
    int max_k = -1;
    for (auto& p : sorted_costs) {
        if (p.second > max_k) {
            max_k = p.second;
            query_lookup.push_back({p.first, max_k});
        }
    }
    
    int Q;
    if (cin >> Q) {
        vector<long long> queries(Q);
        for (int i = 0; i < Q; ++i) cin >> queries[i];
        
        parlay::parallel_for(0, Q, [&](size_t i) {
            long long S = queries[i];
            auto it = upper_bound(query_lookup.begin(), query_lookup.end(), make_pair(S, N + 1));
            int ans = 0;
            if (it != query_lookup.begin()) {
                ans = prev(it)->second;
            }
            queries[i] = ans;
        });
        
        for (int i = 0; i < Q; ++i) {
            cout << queries[i] << "\n";
        }
    }
    
    return 0;
}
//EVOLVE-BLOCK-END