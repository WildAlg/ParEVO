/*
 * Solution for "Little sheep Be" problem.
 * 
 * Algorithm Overview:
 * 1. Graph Construction: CSR-like adjacency list for efficiency.
 * 2. Component Decomposition: Sequential BFS to partition nodes.
 * 3. Parallel Solving: Process components in parallel using Parlay.
 *    - Formulation: r_u = K_u * x + C_u.
 *    - Constraints: r_u >= 0 -> range [L, R] for x.
 *    - Optimization: Minimize sum(r_u^2) -> quadratic minimization.
 *    - Cycle Handling: Consistency checks or fixing x.
 * 
 * Complexity: O(N + M) time and space.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <limits>

#include <parlay/parallel.h>
#include <parlay/sequence.h>

using namespace std;

using Real = double;
const Real EPS = 1e-7;
const Real INF = 1e18;

struct Edge {
    int to;
    Real weight;
};

// CSR-like Graph structure
struct Graph {
    vector<int> head;
    vector<Edge> edges;
    
    void build(int n, const vector<pair<int, int>>& input_edges, const vector<pair<Real, Real>>& coords) {
        vector<int> degree(n, 0);
        for (const auto& e : input_edges) {
            degree[e.first]++;
            degree[e.second]++;
        }
        
        head.assign(n + 1, 0);
        for (int i = 0; i < n; ++i) head[i+1] = head[i] + degree[i];
        
        edges.resize(head[n]);
        vector<int> cur = head;
        
        for (const auto& e : input_edges) {
            int u = e.first;
            int v = e.second;
            Real dx = coords[u].first - coords[v].first;
            Real dy = coords[u].second - coords[v].second;
            Real w = std::sqrt(dx*dx + dy*dy);
            
            edges[cur[u]++] = {v, w};
            edges[cur[v]++] = {u, w};
        }
    }
};

int n, m;
Graph G;
vector<Real> final_radii;
vector<int> K; 
vector<Real> C;
vector<int> comp_nodes;
vector<pair<int, int>> comp_ranges;

bool solve_range(int start_idx, int count) {
    if (count == 0) return true;
    int root = comp_nodes[start_idx];
    
    vector<pair<int, int>> q;
    q.reserve(count);
    q.push_back({root, -1});
    
    K[root] = 1;
    C[root] = 0.0;
    
    bool x_fixed = false;
    Real fixed_val = 0.0;
    Real L = -INF, R = INF;
    Real sum_kc = 0.0;
    
    size_t h = 0;
    while(h < q.size()) {
        auto [u, p] = q[h++];
        
        if (K[u] == 1) {
             if (-C[u] > L) L = -C[u];
        } else {
             if (C[u] < R) R = C[u];
        }
        sum_kc += K[u] * C[u];
        
        int start = G.head[u];
        int end = G.head[u+1];
        
        for (int i = start; i < end; ++i) {
            int v = G.edges[i].to;
            if (v == p) continue;
            
            Real w = G.edges[i].weight;
            
            if (K[v] != 0) {
                int k_sum = K[u] + K[v];
                Real c_sum = C[u] + C[v];
                if (k_sum == 0) {
                    if (std::abs(w - c_sum) > EPS) return false;
                } else {
                    Real val = (w - c_sum) / k_sum;
                    if (x_fixed) {
                        if (std::abs(val - fixed_val) > EPS) return false;
                    } else {
                        x_fixed = true;
                        fixed_val = val;
                    }
                }
            } else {
                K[v] = -K[u];
                C[v] = w - C[u];
                q.push_back({v, u});
            }
        }
    }
    
    if (L > R + EPS) return false;
    
    Real x_final;
    if (x_fixed) {
        if (fixed_val < L - EPS || fixed_val > R + EPS) return false;
        x_final = fixed_val;
    } else {
        Real x_opt = -sum_kc / (Real)count;
        if (x_opt < L) x_final = L;
        else if (x_opt > R) x_final = R;
        else x_final = x_opt;
    }
    
    for (auto [u, p] : q) {
        final_radii[u] = K[u] * x_final + C[u];
    }
    
    return true;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    if (!(cin >> n >> m)) return 0;
    
    vector<pair<Real, Real>> coords(n);
    for(int i=0; i<n; ++i) cin >> coords[i].first >> coords[i].second;
    
    vector<pair<int, int>> edges_in(m);
    for(int i=0; i<m; ++i) {
        cin >> edges_in[i].first >> edges_in[i].second;
        edges_in[i].first--; 
        edges_in[i].second--;
    }
    
    G.build(n, edges_in, coords);
    
    final_radii.resize(n);
    K.assign(n, 0);
    C.resize(n);
    
    vector<char> visited(n, 0);
    comp_nodes.reserve(n);
    
    for(int i=0; i<n; ++i) {
        if(!visited[i]) {
            int start_idx = comp_nodes.size();
            int count = 0;
            vector<int> q; 
            q.reserve(128);
            q.push_back(i);
            visited[i] = 1;
            comp_nodes.push_back(i);
            count++;
            
            size_t h = 0;
            while(h < q.size()) {
                int u = q[h++];
                int s = G.head[u];
                int e = G.head[u+1];
                for(int k=s; k<e; ++k) {
                    int v = G.edges[k].to;
                    if(!visited[v]) {
                        visited[v] = 1;
                        comp_nodes.push_back(v);
                        q.push_back(v);
                        count++;
                    }
                }
            }
            comp_ranges.push_back({start_idx, count});
        }
    }
    
    parlay::sequence<bool> results(comp_ranges.size());
    parlay::parallel_for(0, comp_ranges.size(), [&](size_t i) {
        results[i] = solve_range(comp_ranges[i].first, comp_ranges[i].second);
    });
    
    for(bool r : results) {
        if(!r) {
            cout << "NE\n";
            return 0;
        }
    }
    
    cout << "DA\n";
    cout << fixed << setprecision(6);
    for(int i=0; i<n; ++i) cout << final_radii[i] << "\n";
    
    return 0;
}