//EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>

#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

using namespace std;

const long long INF = 1e16;

// Structure to hold DP states for a subtree
// Flattened representation for better cache locality
// Layout: s0... s1... s2...
// Each segment has size (sz + 1)
using DPState = vector<long long>;

struct TreeResult {
    DPState dp;
    int sz;
};

// Helper to update minimum value
inline void update_min(long long& target, long long val) {
    if (val < target) target = val;
}

// Merges child's DP state into parent's DP state
void merge_child(TreeResult& parent, const TreeResult& child) {
    int p_sz = parent.sz;
    int c_sz = child.sz;
    int new_sz = p_sz + c_sz;
    
    int p_w = p_sz + 1;
    int c_w = c_sz + 1;
    int n_w = new_sz + 1;
    
    DPState next_dp(3 * n_w, INF);
    
    // Pointers for direct access
    const long long* p = parent.dp.data();
    const long long* c = child.dp.data();
    long long* n = next_dp.data();
    
    const long long* p0 = p;
    const long long* p1 = p + p_w;
    const long long* p2 = p + 2 * p_w;
    
    const long long* c0 = c;
    const long long* c1 = c + c_w;
    const long long* c2 = c + 2 * c_w;
    
    long long* n0 = n;
    long long* n1 = n + n_w;
    long long* n2 = n + 2 * n_w;
    
    // Parent State 0: u not relaxed
    for (int i = 0; i <= p_sz; ++i) {
        if (p0[i] == INF) continue;
        long long val = p0[i];
        for (int j = 0; j <= c_sz; ++j) {
            // Child 0 or 2
            long long cost = c0[j];
            if (c2[j] < cost) cost = c2[j];
            
            if (cost != INF) {
                if (val + cost < n0[i+j]) n0[i+j] = val + cost;
            }
        }
    }
    
    // Parent State 1: u relaxed, isolated
    for (int i = 0; i <= p_sz; ++i) {
        if (p1[i] == INF) continue;
        long long val = p1[i];
        for (int j = 0; j <= c_sz; ++j) {
            // Child 0 -> Parent 1
            if (c0[j] != INF) {
                if (val + c0[j] < n1[i+j]) n1[i+j] = val + c0[j];
            }
            // Child 1 -> Parent 2 (both satisfied, +2 count)
            if (c1[j] != INF) {
                if (val + c1[j] < n2[i+j+2]) n2[i+j+2] = val + c1[j];
            }
            // Child 2 -> Parent 2 (parent satisfied, +1 count)
            if (c2[j] != INF) {
                if (val + c2[j] < n2[i+j+1]) n2[i+j+1] = val + c2[j];
            }
        }
    }
    
    // Parent State 2: u relaxed, satisfied
    for (int i = 0; i <= p_sz; ++i) {
        if (p2[i] == INF) continue;
        long long val = p2[i];
        for (int j = 0; j <= c_sz; ++j) {
            // Child 0 -> Parent 2
            if (c0[j] != INF) {
                if (val + c0[j] < n2[i+j]) n2[i+j] = val + c0[j];
            }
            // Child 1 -> Parent 2 (child satisfied, +1 count)
            if (c1[j] != INF) {
                if (val + c1[j] < n2[i+j+1]) n2[i+j+1] = val + c1[j];
            }
            // Child 2 -> Parent 2 (both satisfied, +0 count)
            if (c2[j] != INF) {
                if (val + c2[j] < n2[i+j]) n2[i+j] = val + c2[j];
            }
        }
    }
    
    parent.dp = move(next_dp);
    parent.sz = new_sz;
}

struct Node {
    long long d;
    vector<int> adj;
};
vector<Node> nodes;

TreeResult solve_tree(int u, int p) {
    TreeResult res;
    res.sz = 1;
    res.dp.assign(6, INF);
    // s0[0] = 0
    res.dp[0] = 0;
    // s1[0] = D[u] (index 2)
    res.dp[2] = nodes[u].d;
    // s2 is INF
    
    for (int v : nodes[u].adj) {
        if (v == p) continue;
        TreeResult child = solve_tree(v, u);
        merge_child(res, child);
    }
    return res;
}

// Global merge of two component results
vector<long long> merge_components(const vector<long long>& A, const vector<long long>& B) {
    if (A.size() == 1 && A[0] == 0) return B;
    if (B.size() == 1 && B[0] == 0) return A;
    
    int sz_a = A.size() - 1;
    int sz_b = B.size() - 1;
    vector<long long> C(sz_a + sz_b + 1, INF);
    
    for (int i = 0; i <= sz_a; ++i) {
        if (A[i] == INF) continue;
        long long val_a = A[i];
        for (int j = 0; j <= sz_b; ++j) {
            if (B[j] != INF) {
                if (val_a + B[j] < C[i+j]) C[i+j] = val_a + B[j];
            }
        }
    }
    return C;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int N, M;
    if (!(cin >> N >> M)) return 0;
    
    nodes.resize(N + 1);
    for (int i = 1; i <= N; ++i) {
        cin >> nodes[i].d;
    }
    
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        nodes[u].adj.push_back(v);
        nodes[v].adj.push_back(u);
    }
    
    vector<bool> visited(N + 1, false);
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
        TreeResult res = solve_tree(root, 0);
        int sz = res.sz;
        int w = sz + 1;
        vector<long long> costs(sz + 1);
        for (int k = 0; k <= sz; ++k) {
            costs[k] = res.dp[k]; // s0
            if (res.dp[2*w + k] < costs[k]) costs[k] = res.dp[2*w + k]; // s2
        }
        return costs;
    });
    
    vector<long long> global_dp = parlay::reduce(tree_results, parlay::binary_op(merge_components, vector<long long>{0}));
    
    vector<pair<long long, int>> sorted_costs;
    sorted_costs.reserve(global_dp.size());
    for (int k = 0; k < global_dp.size(); ++k) {
        if (global_dp[k] != INF) {
            sorted_costs.push_back({global_dp[k], k});
        }
    }
    sort(sorted_costs.begin(), sorted_costs.end());
    
    vector<pair<long long, int>> query_lookup;
    int max_k = -1;
    for (const auto& p : sorted_costs) {
        if (p.second > max_k) {
            max_k = p.second;
            query_lookup.push_back({p.first, max_k});
        }
    }
    
    int Q;
    if (cin >> Q) {
        vector<long long> queries(Q);
        for (int i = 0; i < Q; ++i) cin >> queries[i];
        
        vector<int> answers(Q);
        parlay::parallel_for(0, Q, [&](size_t i) {
            long long S = queries[i];
            auto it = upper_bound(query_lookup.begin(), query_lookup.end(), make_pair(S, N + 1));
            int ans = 0;
            if (it != query_lookup.begin()) {
                ans = prev(it)->second;
            }
            answers[i] = ans;
        });
        
        for (int i = 0; i < Q; ++i) {
            cout << answers[i] << "\n";
        }
    }
    
    return 0;
}
//EVOLVE-BLOCK-END