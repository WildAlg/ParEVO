//EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Iterative DSU with path compression and union by size
struct DSU {
    vector<int> parent;
    vector<int> sz;
    vector<int> edges;

    DSU(int n) {
        parent.resize(n + 1);
        // iota is in <numeric>
        iota(parent.begin(), parent.end(), 0);
        sz.assign(n + 1, 1);
        edges.assign(n + 1, 0);
    }

    int find(int i) {
        int root = i;
        while (parent[root] != root) {
            root = parent[root];
        }
        // Path compression
        int curr = i;
        while (curr != root) {
            int next = parent[curr];
            parent[curr] = root;
            curr = next;
        }
        return root;
    }

    void unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            if (sz[root_i] < sz[root_j])
                swap(root_i, root_j);
            parent[root_j] = root_i;
            sz[root_i] += sz[root_j];
            edges[root_i] += edges[root_j] + 1;
        } else {
            edges[root_i]++;
        }
    }
};

struct Result {
    long long students;
    long long cuts;
};

int main() {
    // Fast IO
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    long long K;
    if (!(cin >> N >> M >> K)) return 0;

    DSU dsu(N);

    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        dsu.unite(u, v);
    }

    // Identify roots.
    // We create a sequence of indices and filter for roots.
    // parlay::tabulate creates a sequence 1..N
    auto nodes = parlay::tabulate(N, [](size_t i) { return (int)(i + 1); });
    
    // Filter to get only roots
    auto roots = parlay::filter(nodes, [&](int i) {
        return dsu.parent[i] == i;
    });

    // Map each root to a Result (students, cuts) and Reduce to sum them up
    // We use a custom monoid to sum the results
    auto result_monoid = parlay::make_monoid([](Result a, Result b) {
        return Result{a.students + b.students, a.cuts + b.cuts};
    }, Result{0, 0});

    Result total = parlay::reduce(
        parlay::map(roots, [&](int r) -> Result {
            long long L = dsu.sz[r];
            long long E = dsu.edges[r];
            
            // A component is a cycle if Edges == Vertices
            // It is a path if Edges == Vertices - 1
            // Max degree 2 ensures these are the only possibilities for connected components
            bool is_cycle = (E == L);
            
            long long s = 0;
            long long c = 0;

            if (L >= K) {
                // We can form L/K groups of size K
                s = (L / K) * K;
                
                if (is_cycle) {
                    if (L == K) {
                        // Perfect cycle, take all, 0 cuts
                        c = 0;
                    } else {
                        // Cycle > K
                        // Must cut at least 1 edge to break cycle
                        // If L % K == 0, we can cut into exact pieces
                        // e.g. L=4, K=2 -> cut 2 edges -> 2 pieces. C = L/K
                        // e.g. L=6, K=2 -> cut 3 edges -> 3 pieces. C = L/K
                        // If L % K != 0, we have remainder
                        // e.g. L=5, K=2 -> cut 3 edges -> 2 pieces + remainder. C = L/K + 1
                        if (L % K == 0) {
                            c = L / K;
                        } else {
                            c = L / K + 1;
                        }
                    }
                } else {
                    // Path
                    // If L % K == 0, we cut L/K - 1 edges
                    // e.g. L=4, K=2 -> cut 1 edge.
                    // If L % K != 0, we cut L/K edges (last piece is remainder)
                    // e.g. L=5, K=2 -> cut 2 edges.
                    if (L % K == 0) {
                        c = L / K - 1;
                    } else {
                        c = L / K;
                    }
                }
            }
            return {s, c};
        }),
        result_monoid
    );

    cout << total.students << " " << total.cuts << endl;

    return 0;
}
//EVOLVE-BLOCK-END