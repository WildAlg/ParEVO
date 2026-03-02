/**
 * Canadian Computing Competition: 2008 Stage 2, Day 1, Problem 2
 * Problem: Streets
 * 
 * Solution Overview:
 * 1. Efficient I/O: Reads the entire input into a large memory buffer to minimize I/O syscall overhead.
 *    Uses `std::string_view` to reference street names directly from the buffer, avoiding string copies.
 * 2. Parallel Processing (ParlayLib):
 *    - Sorts and deduplicates street names in parallel to map them to unique integer IDs.
 *    - Maps observations and queries to IDs in parallel using binary search on the sorted names.
 *    - Processes queries in parallel after building the connectivity graph.
 * 3. Algorithm:
 *    - Uses Disjoint Set Union (DSU) with parity tracking (path compression and union by rank/index)
 *      to manage connected components and relative street orientations.
 *    - DSU state tracks the relationship of each node to its parent (0: parallel, 1: perpendicular).
 *    - If a contradiction is found during observation processing, outputs "Waterloo".
 *    - Before query processing, the DSU is flattened to ensure O(1) read-only access during the parallel query phase.
 */

#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <algorithm>
#include <iterator>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// Disjoint Set Union (DSU) with parity tracking.
// parent[i]: Parent of node i.
// rel[i]: Relationship of i relative to parent[i] (0: parallel, 1: perpendicular).
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
        
        // We need: orientation(i) ^ orientation(j) = type
        // Derived: rel(ri, rj) = type ^ rel(i) ^ rel(j)
        int needed_rel = type ^ root_i.second ^ root_j.second;

        if (ri != rj) {
            // Merge tree rooted at ri into rj.
            parent[ri] = rj;
            rel[ri] = needed_rel;
            return true;
        } else {
            // Already in the same component, check for consistency.
            // If roots are same, relative relation must be 0 (identity).
            return (needed_rel == 0);
        }
    }
};

int main() {
    // Optimize standard I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    // Read entire input into a buffer to avoid repetitive I/O and string allocations
    string buffer;
    // Reserve memory to reduce reallocations (heuristic size for large inputs)
    buffer.reserve(1 << 24); 
    buffer.assign(istreambuf_iterator<char>(cin), istreambuf_iterator<char>());

    if (buffer.empty()) return 0;

    // Tokenizer setup
    const char* str = buffer.data();
    size_t len = buffer.size();
    size_t pos = 0;

    // Fast token extraction using string_view
    auto skip_ws = [&]() {
        while (pos < len && str[pos] <= 32) pos++;
    };

    auto read_token = [&]() -> string_view {
        skip_ws();
        if (pos == len) return {};
        size_t start = pos;
        while (pos < len && str[pos] > 32) pos++;
        return string_view(str + start, pos - start);
    };

    // Fast integer parsing
    auto to_int = [](string_view s) {
        int res = 0;
        for (char c : s) res = res * 10 + (c - '0');
        return res;
    };

    string_view m_sv = read_token();
    string_view n_sv = read_token();
    
    if (m_sv.empty()) return 0;

    int m = to_int(m_sv);
    int n = to_int(n_sv);

    // Structures to hold parsed but unmapped data
    struct RawObs { string_view u, v; int type; };
    vector<RawObs> raw_obs(m);
    
    // Collect all street names for mapping
    vector<string_view> all_names;
    all_names.reserve(2 * m + 2 * n);

    for(int i = 0; i < m; ++i) {
        raw_obs[i].u = read_token();
        raw_obs[i].v = read_token();
        string_view t = read_token();
        // "intersect" starts with 'i', "parallel" with 'p'
        raw_obs[i].type = (t[0] == 'i' ? 1 : 0);
        all_names.push_back(raw_obs[i].u);
        all_names.push_back(raw_obs[i].v);
    }

    struct RawQuery { string_view u, v; };
    vector<RawQuery> raw_queries(n);
    for(int i = 0; i < n; ++i) {
        raw_queries[i].u = read_token();
        raw_queries[i].v = read_token();
        all_names.push_back(raw_queries[i].u);
        all_names.push_back(raw_queries[i].v);
    }

    // Convert to parlay sequence for parallel sorting
    parlay::sequence<string_view> names(all_names.begin(), all_names.end());
    
    // Sort names in parallel
    parlay::sort_inplace(names);
    
    // Create distinct names sequence using parlay::pack
    // Identify boundaries where names[i] != names[i-1]
    auto flags = parlay::delayed_seq<bool>(names.size(), [&](size_t i) {
        return (i == 0) || (names[i] != names[i-1]);
    });
    auto distinct_names = parlay::pack(names, flags);

    // Lambda to find ID of a name using binary search
    auto get_id = [&](string_view s) -> int {
        auto it = std::lower_bound(distinct_names.begin(), distinct_names.end(), s);
        return (int)(it - distinct_names.begin());
    };

    // Convert observations to IDs in parallel
    struct Obs { int u, v, type; };
    parlay::sequence<Obs> obs(m);
    parlay::parallel_for(0, m, [&](size_t i) {
        obs[i] = {get_id(raw_obs[i].u), get_id(raw_obs[i].v), raw_obs[i].type};
    });

    // Process observations using DSU (sequential due to dependencies)
    DSU dsu(distinct_names.size());
    for(int i = 0; i < m; ++i) {
        if (!dsu.unite(obs[i].u, obs[i].v, obs[i].type)) {
            cout << "Waterloo\n";
            return 0;
        }
    }

    // Flatten the DSU structure so that all nodes point directly to their root.
    // This enables safe, read-only O(1) access during the parallel query phase.
    for(int i = 0; i < (int)distinct_names.size(); ++i) {
        dsu.find(i);
    }

    // Process queries in parallel
    // Results: 0 -> parallel, 1 -> intersect, 2 -> unknown
    parlay::sequence<int> results(n);
    parlay::parallel_for(0, n, [&](size_t i) {
        int u = get_id(raw_queries[i].u);
        int v = get_id(raw_queries[i].v);
        
        int root_u = dsu.parent[u];
        int root_v = dsu.parent[v];
        
        if (root_u != root_v) {
            results[i] = 2; // unknown (different components)
        } else {
            // Same component: check relative orientation
            int rel = dsu.rel[u] ^ dsu.rel[v];
            results[i] = (rel == 0 ? 0 : 1);
        }
    });

    // Output results
    const char* answers[] = {"parallel", "intersect", "unknown"};
    for(int i = 0; i < n; ++i) {
        cout << answers[results[i]] << "\n";
    }

    return 0;
}