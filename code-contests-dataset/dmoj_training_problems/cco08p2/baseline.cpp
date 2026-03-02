#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Disjoint Set Union (DSU) with parity to track relative orientations.
// parent[i] stores the parent of node i.
// rel[i] stores the relationship of i relative to parent[i]: 0 for parallel, 1 for perpendicular.
struct DSU {
    vector<int> parent;
    vector<int> rel;

    DSU(int n) {
        parent.resize(n);
        rel.assign(n, 0);
        for(int i = 0; i < n; ++i) parent[i] = i;
    }

    // Find with path compression.
    // Returns {root, parity relative to root}.
    pair<int, int> find(int i) {
        if (parent[i] != i) {
            pair<int, int> root_info = find(parent[i]);
            parent[i] = root_info.first;
            rel[i] = rel[i] ^ root_info.second;
        }
        return {parent[i], rel[i]};
    }

    // Unite two sets. Returns false if a contradiction is found.
    // type: 0 for parallel, 1 for intersect.
    bool unite(int i, int j, int type) {
        pair<int, int> root_i = find(i);
        pair<int, int> root_j = find(j);

        int ri = root_i.first;
        int rj = root_j.first;
        int pi = root_i.second;
        int pj = root_j.second;

        if (ri != rj) {
            // Merge tree rooted at ri into rj.
            // We need: orientation(i) ^ orientation(j) = type
            // (orientation(ri) ^ pi) ^ (orientation(rj) ^ pj) = type
            // orientation(ri) ^ orientation(rj) = type ^ pi ^ pj
            parent[ri] = rj;
            rel[ri] = pi ^ pj ^ type;
            return true;
        } else {
            // Already in the same component, check for consistency.
            return (pi ^ pj) == type;
        }
    }
};

struct QueryID {
    int u, v;
};

int main() {
    // Optimize I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int m, n;
    if (!(cin >> m >> n)) return 0;

    // Map street names to integer IDs
    unordered_map<string, int> name_to_id;
    name_to_id.reserve((m + n) * 2);
    int id_counter = 0;

    auto get_id = [&](const string& s) {
        auto it = name_to_id.find(s);
        if (it == name_to_id.end()) {
            name_to_id[s] = id_counter;
            return id_counter++;
        }
        return it->second;
    };

    // Read and store observations
    struct Obs { int u, v, type; };
    vector<Obs> observations(m);

    for(int i = 0; i < m; ++i) {
        string u, v, t;
        cin >> u >> v >> t;
        observations[i] = {get_id(u), get_id(v), (t == "intersect" ? 1 : 0)};
    }

    // Read queries into a parlay sequence for parallel processing
    parlay::sequence<QueryID> queries(n);
    for(int i = 0; i < n; ++i) {
        string u, v;
        cin >> u >> v;
        queries[i] = {get_id(u), get_id(v)};
    }

    // Process observations using DSU
    DSU dsu(id_counter);
    for(const auto& obs : observations) {
        if (!dsu.unite(obs.u, obs.v, obs.type)) {
            cout << "Waterloo" << endl;
            return 0;
        }
    }

    // Flatten the DSU structure so that all nodes point directly to their root.
    // This allows O(1) read-only access during the parallel query phase.
    for(int i = 0; i < id_counter; ++i) {
        dsu.find(i);
    }

    // Process queries in parallel using parlay::map
    // Results: 0 -> parallel, 1 -> unknown, 2 -> intersect
    auto results = parlay::map(queries, [&](const QueryID& q) {
        int root_u = dsu.parent[q.u];
        int root_v = dsu.parent[q.v];

        // If roots are different, streets are in disconnected components.
        // Their relative orientation is not fixed.
        if (root_u != root_v) return 1;

        // If roots are same, check relative parity.
        int par_u = dsu.rel[q.u];
        int par_v = dsu.rel[q.v];

        if (par_u == par_v) return 0; // Same orientation -> parallel
        else return 2; // Different orientation -> intersect
    });

    // Output results
    const char* answers[] = {"parallel", "unknown", "intersect"};
    for(int r : results) {
        cout << answers[r] << "\n";
    }

    return 0;
}