/**
 * Problem: Little Mirko's Coloring Book
 * Approach: Functional Graph Decomposition
 * 
 * The problem is modeled as a functional graph where each node i has a directed edge to f[i].
 * The graph consists of several components, each containing exactly one cycle and a set of trees rooted on the cycle nodes.
 * 
 * 1. Tree Nodes:
 *    Any node that is part of a tree (not in a cycle) must have a color different from its parent (f[i]).
 *    However, the coloring dependency flows from f[i] to i. If f[i] is colored, there are K-1 choices for i.
 *    We can count these nodes by topologically sorting (peeling) the graph starting from nodes with in-degree 0.
 *    Total ways for tree nodes = (K-1)^(number of tree nodes).
 * 
 * 2. Cycle Nodes:
 *    After peeling, only cycles remain.
 *    For a cycle of length L:
 *    - If L = 1 (f[i] = i), the constraint is lifted. There are K ways to color this node.
 *    - If L > 1, we must color L nodes such that adjacent nodes have different colors.
 *      The number of ways is given by the chromatic polynomial of a cycle C_L:
 *      Ways = (K-1)^L + (-1)^L * (K-1).
 * 
 * The final answer is the product of ways for all components modulo 10^9 + 7.
 */

#include <iostream>
#include <vector>
#include <algorithm>

#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

const int MOD = 1000000007;

// Fast I/O
void fast_io() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
}

// Modular exponentiation
long long power(long long base, long long exp) {
    long long res = 1;
    base %= MOD;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % MOD;
        base = (base * base) % MOD;
        exp /= 2;
    }
    return res;
}

int main() {
    fast_io();

    int N;
    long long K;
    if (!(cin >> N >> K)) return 0;

    // Use parlay sequences for storage
    // f[i] is the node that node i depends on
    parlay::sequence<int> f(N + 1);
    parlay::sequence<int> in_degree(N + 1, 0);

    for (int i = 1; i <= N; ++i) {
        cin >> f[i];
        in_degree[f[i]]++;
    }

    // Identify initial leaves (nodes with in-degree 0) using parlay's parallel filter
    // parlay::iota(N + 1) generates 0, 1, ..., N
    auto leaves = parlay::filter(parlay::iota(N + 1), [&](int i) {
        return i > 0 && in_degree[i] == 0;
    });

    // Queue for topological sort (peeling)
    vector<int> q;
    q.reserve(N);
    for (int u : leaves) {
        q.push_back(u);
    }

    // Process the queue to peel off tree nodes
    int head = 0;
    while(head < q.size()) {
        int u = q[head++];
        int v = f[u];
        in_degree[v]--;
        if (in_degree[v] == 0) {
            q.push_back(v);
        }
    }

    // Each tree node contributes (K-1) choices
    long long tree_nodes = q.size();
    long long total_ways = power(K - 1, tree_nodes);

    // Process remaining nodes (cycles)
    // After peeling, nodes with in_degree > 0 are exactly the nodes in cycles
    for (int i = 1; i <= N; ++i) {
        if (in_degree[i] > 0) {
            int curr = i;
            long long len = 0;
            
            // Traverse the cycle to determine length and mark as visited
            while (in_degree[curr] > 0) {
                in_degree[curr] = 0; // Mark as visited by setting in-degree to 0
                curr = f[curr];
                len++;
            }
            
            long long ways = 0;
            if (len == 1) {
                // Self-loop: f[i] = i. Restriction is lifted.
                ways = K % MOD;
            } else {
                // Cycle of length L > 1
                // Ways = (K-1)^L + (-1)^L * (K-1)
                long long term1 = power(K - 1, len);
                long long term2 = (K - 1) % MOD;
                
                if (len % 2 == 1) {
                    ways = (term1 - term2 + MOD) % MOD;
                } else {
                    ways = (term1 + term2) % MOD;
                }
            }
            
            total_ways = (total_ways * ways) % MOD;
        }
    }

    cout << total_ways << endl;

    return 0;
}