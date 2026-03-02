#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <array>
#include <limits>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/parallel.h>

// Type aliases for clarity
using ll = long long;
using dp_type = std::vector<ll>;
const ll INF = std::numeric_limits<ll>::max();

// Helper to trim trailing INFs from a DP vector
void trim(dp_type& v) {
    size_t new_size = v.size();
    while (new_size > 0 && v[new_size - 1] == INF) {
        new_size--;
    }
    v.resize(new_size);
}

// Merges source vector into target vector, keeping min cost for each count.
void merge_into(dp_type& target, const dp_type& source) {
    if (source.empty()) return;
    if (target.empty()) {
        target = source;
        return;
    }
    if (target.size() < source.size()) {
        target.resize(source.size(), INF);
    }
    for (size_t i = 0; i < source.size(); ++i) {
        if (source[i] != INF) {
            target[i] = std::min(target[i], source[i]);
        }
    }
    trim(target);
}

// (min, +) convolution of two DP vectors.
dp_type convolve(const dp_type& v1, const dp_type& v2) {
    if (v1.empty() || v2.empty()) return {};
    
    const dp_type& small_v = (v1.size() < v2.size()) ? v1 : v2;
    const dp_type& large_v = (v1.size() < v2.size()) ? v2 : v1;
    
    dp_type res(small_v.size() + large_v.size() - 1, INF);
    for (size_t i = 0; i < small_v.size(); ++i) {
        if (small_v[i] == INF) continue;
        for (size_t j = 0; j < large_v.size(); ++j) {
            if (large_v[j] != INF) {
                res[i+j] = std::min(res[i+j], small_v[i] + large_v[j]);
            }
        }
    }
    trim(res);
    return res;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int N, M;
    std::cin >> N >> M;

    parlay::sequence<ll> D(N);
    for (int i = 0; i < N; ++i) std::cin >> D[i];

    parlay::sequence<parlay::sequence<int>> adj(N);
    for (int i = 0; i < M; ++i) {
        int u, v;
        std::cin >> u >> v;
        --u; --v; // 0-indexed
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Find connected components using iterative BFS
    parlay::sequence<int> component_map(N, -1);
    parlay::sequence<parlay::sequence<int>> components;
    int component_count = 0;
    for (int i = 0; i < N; ++i) {
        if (component_map[i] == -1) {
            parlay::sequence<int> q;
            q.push_back(i);
            component_map[i] = component_count;
            parlay::sequence<int> current_comp_nodes;
            size_t head = 0;
            while(head < q.size()){
                int u = q[head++];
                current_comp_nodes.push_back(u);
                for(int v : adj[u]){
                    if(component_map[v] == -1){
                        component_map[v] = component_count;
                        q.push_back(v);
                    }
                }
            }
            components.push_back(std::move(current_comp_nodes));
            component_count++;
        }
    }

    // DP for each component (tree) in parallel
    parlay::sequence<dp_type> tree_dps(component_count);
    
    parlay::parallel_for(0, component_count, [&](size_t i) {
        auto& component_nodes = components[i];
        if (component_nodes.size() <= 1) {
            tree_dps[i] = {0};
            return;
        }

        int root = component_nodes[0];
        
        std::vector<std::vector<int>> children(N);
        std::vector<int> parent(N, -1);
        std::vector<int> q_bfs;
        
        q_bfs.push_back(root);
        parent[root] = -2; // Mark root as visited
        
        size_t head = 0;
        while(head < q_bfs.size()){
            int u = q_bfs[head++];
            for(int v : adj[u]){
                if(component_map[v] == (int)i && parent[v] == -1){
                    parent[v] = u;
                    children[u].push_back(v);
                    q_bfs.push_back(v);
                }
            }
        }
        
        std::reverse(q_bfs.begin(), q_bfs.end()); // post-order traversal
        
        std::vector<std::array<dp_type, 3>> node_dps(N);

        for (int u : q_bfs) {
            dp_type cur_dp0 = {0};
            dp_type cur_dp1 = {D[u]};
            dp_type cur_dp2;

            for (int v : children[u]) {
                auto& child_dp = node_dps[v];
                
                dp_type next_dp0, next_dp1, next_dp2;

                // State 0: u not relaxed. Child v can be in any state.
                dp_type child_any = child_dp[0];
                merge_into(child_any, child_dp[1]);
                merge_into(child_any, child_dp[2]);
                next_dp0 = convolve(cur_dp0, child_any);

                // State 1: u relaxed, not exchanging. Child v must not be relaxed.
                next_dp1 = convolve(cur_dp1, child_dp[0]);

                // State 2: u relaxed and exchanging. Optimized calculation.
                size_t sz_res = 0;
                auto get_sz = [](const auto& v1, const auto& v2, int add_k){
                    if (v1.empty() || v2.empty()) return (size_t)0;
                    return v1.size() + v2.size() - 1 + add_k;
                };
                sz_res = std::max(sz_res, get_sz(cur_dp1, child_dp[1], 2));
                sz_res = std::max(sz_res, get_sz(cur_dp1, child_dp[2], 1));
                if (!cur_dp2.empty()) {
                    sz_res = std::max(sz_res, get_sz(cur_dp2, child_dp[0], 0));
                    sz_res = std::max(sz_res, get_sz(cur_dp2, child_dp[1], 1));
                    sz_res = std::max(sz_res, get_sz(cur_dp2, child_dp[2], 0));
                }

                if (sz_res > 0) {
                    next_dp2.assign(sz_res, INF);
                    // Case B: u becomes exchanging
                    if (!cur_dp1.empty()) {
                        if (!child_dp[1].empty())
                            for (size_t i1 = 0; i1 < cur_dp1.size(); ++i1) if (cur_dp1[i1] != INF)
                                for (size_t j1 = 0; j1 < child_dp[1].size(); ++j1) if (child_dp[1][j1] != INF)
                                    next_dp2[i1+j1+2] = std::min(next_dp2[i1+j1+2], cur_dp1[i1] + child_dp[1][j1]);
                        if (!child_dp[2].empty())
                            for (size_t i1 = 0; i1 < cur_dp1.size(); ++i1) if (cur_dp1[i1] != INF)
                                for (size_t j1 = 0; j1 < child_dp[2].size(); ++j1) if (child_dp[2][j1] != INF)
                                    next_dp2[i1+j1+1] = std::min(next_dp2[i1+j1+1], cur_dp1[i1] + child_dp[2][j1]);
                    }
                    // Case A: u was already exchanging
                    if (!cur_dp2.empty()) {
                        if (!child_dp[0].empty())
                            for (size_t i2 = 0; i2 < cur_dp2.size(); ++i2) if (cur_dp2[i2] != INF)
                                for (size_t j2 = 0; j2 < child_dp[0].size(); ++j2) if (child_dp[0][j2] != INF)
                                    next_dp2[i2+j2] = std::min(next_dp2[i2+j2], cur_dp2[i2] + child_dp[0][j2]);
                        if (!child_dp[1].empty())
                            for (size_t i2 = 0; i2 < cur_dp2.size(); ++i2) if (cur_dp2[i2] != INF)
                                for (size_t j2 = 0; j2 < child_dp[1].size(); ++j2) if (child_dp[1][j2] != INF)
                                    next_dp2[i2+j2+1] = std::min(next_dp2[i2+j2+1], cur_dp2[i2] + child_dp[1][j2]);
                        if (!child_dp[2].empty())
                            for (size_t i2 = 0; i2 < cur_dp2.size(); ++i2) if (cur_dp2[i2] != INF)
                                for (size_t j2 = 0; j2 < child_dp[2].size(); ++j2) if (child_dp[2][j2] != INF)
                                    next_dp2[i2+j2] = std::min(next_dp2[i2+j2], cur_dp2[i2] + child_dp[2][j2]);
                    }
                    trim(next_dp2);
                }
                
                cur_dp0 = std::move(next_dp0);
                cur_dp1 = std::move(next_dp1);
                cur_dp2 = std::move(next_dp2);
            }
            node_dps[u] = {std::move(cur_dp0), std::move(cur_dp1), std::move(cur_dp2)};
        }
        
        auto& root_dp = node_dps[root];
        dp_type res = root_dp[0];
        merge_into(res, root_dp[1]);
        merge_into(res, root_dp[2]);
        tree_dps[i] = std::move(res);
    });

    dp_type identity_dp = {0};
    auto convolve_monoid = parlay::make_monoid(convolve, identity_dp);
    dp_type final_dp = parlay::reduce(tree_dps, convolve_monoid);

    parlay::sequence<std::pair<ll, int>> answer_pairs;
    if (!final_dp.empty()) {
        answer_pairs.reserve(final_dp.size());
        for (size_t i = 0; i < final_dp.size(); ++i) {
            if (final_dp[i] != INF) {
                answer_pairs.push_back({final_dp[i], static_cast<int>(i)});
            }
        }
    }
    
    parlay::sort_inplace(answer_pairs);

    if (answer_pairs.size() > 1) {
        // This is sequential but fast enough
        for (size_t i = 1; i < answer_pairs.size(); ++i) {
            answer_pairs[i].second = std::max(answer_pairs[i].second, answer_pairs[i-1].second);
        }
    }

    int Q;
    std::cin >> Q;
    parlay::sequence<ll> S(Q);
    for(int i=0; i<Q; ++i) std::cin >> S[i];

    parlay::sequence<int> results(Q);
    parlay::parallel_for(0, Q, [&](size_t i){
        auto it = std::upper_bound(answer_pairs.begin(), answer_pairs.end(), std::make_pair(S[i], N + 1));
        if (it == answer_pairs.begin()) {
            results[i] = 0;
        } else {
            results[i] = std::prev(it)->second;
        }
    });

    for (int i = 0; i < Q; ++i) {
        std::cout << results[i] << "\n";
    }

    return 0;
}