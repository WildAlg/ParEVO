//EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <algorithm>

#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Fast I/O setup to handle large input efficiently
void fast_io() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
}

// Modular exponentiation to calculate (base^exp) % MOD
long long power(long long base, long long exp) {
    long long res = 1;
    base %= 1000000007;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % 1000000007;
        base = (base * base) % 1000000007;
        exp /= 2;
    }
    return res;
}

int main() {
    fast_io();

    int N;
    long long K;
    if (!(cin >> N >> K)) return 0;

    // Use parlay sequences for data storage
    // f stores the dependency graph where edge is i -> f[i]
    parlay::sequence<int> f(N + 1);
    parlay::sequence<int> in_degree(N + 1, 0);

    // Read input and calculate in-degrees
    for (int i = 1; i <= N; ++i) {
        cin >> f[i];
        in_degree[f[i]]++;
    }

    // Use parlay to filter nodes that have in-degree 0 initially
    // These are the leaves of the tree components (start of dependency chains)
    auto initial_zeros = parlay::filter(parlay::iota(N + 1), [&](size_t i) {
        return i > 0 && in_degree[i] == 0;
    });

    // Queue for peeling process (topological sort style)
    vector<int> q;
    q.reserve(N);
    for (auto x : initial_zeros) {
        q.push_back(x);
    }

    // Process the queue to remove all tree nodes
    // This reduces the functional graph to just its cycles
    int head = 0;
    while(head < q.size()){
        int u = q[head++];
        int v = f[u];
        in_degree[v]--;
        if (in_degree[v] == 0) {
            q.push_back(v);
        }
    }

    // Number of nodes in the trees
    long long tree_nodes = q.size();
    
    // Each tree node contributes (K-1) choices relative to its parent (the node it points to)
    // because c(u) != c(f(u)), so given c(f(u)), there are K-1 choices for c(u).
    long long total_ways = power(K - 1, tree_nodes);

    // Identify nodes that are part of cycles (those not peeled)
    auto cycle_candidates = parlay::filter(parlay::iota(N + 1), [&](size_t i) {
        return i > 0 && in_degree[i] > 0;
    });

    parlay::sequence<bool> visited(N + 1, false);

    // Process each cycle
    for (auto i : cycle_candidates) {
        if (!visited[i]) {
            int curr = i;
            long long len = 0;
            // Traverse the cycle to find its length and mark nodes as visited
            while (!visited[curr]) {
                visited[curr] = true;
                curr = f[curr];
                len++;
            }
            
            long long ways = 0;
            if (len == 1) {
                // Self-loop: f_i = i. The problem states "except when f_i = i".
                // In this case, there is no restriction on node i relative to f_i.
                // It can be any of the K colors.
                ways = K % 1000000007;
            } else {
                // Cycle of length > 1.
                // The number of ways to color a cycle of length L with K colors such that
                // adjacent nodes have different colors is given by the chromatic polynomial of C_L:
                // P(C_L, K) = (K-1)^L + (-1)^L * (K-1)
                long long term1 = power(K - 1, len);
                long long term2 = (K - 1) % 1000000007;
                if (len % 2 == 1) {
                    ways = (term1 - term2 + 1000000007) % 1000000007;
                } else {
                    ways = (term1 + term2) % 1000000007;
                }
            }
            total_ways = (total_ways * ways) % 1000000007;
        }
    }

    cout << total_ways << endl;

    return 0;
}
//EVOLVE-BLOCK-END