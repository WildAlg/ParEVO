/**
 * Canadian Computing Competition: 2008 Stage 2, Day 1, Problem 2
 * Problem: Streets
 * 
 * Solution Overview:
 * 1. Fast I/O: Reads entire input into a memory buffer to minimize system calls.
 * 2. Optimized Mapping: 
 *    - Parses street names into `Item` structs (hash + ptr + len).
 *    - Uses `parlay::sort_inplace` for parallel sorting of street names.
 *    - Assigns IDs via linear scan of the sorted items.
 * 3. Algorithm:
 *    - Uses Disjoint Set Union (DSU) with parity tracking for relative orientations.
 *    - Builds the DSU sequentially to detect "Waterloo" contradictions.
 *    - Flattens the DSU for O(1) query access.
 * 4. Parallel Queries & Output:
 *    - Queries are processed in parallel.
 *    - Output buffer is constructed in parallel using `parlay::scan` and `parlay::parallel_for` 
 *      to calculate offsets and copy strings concurrently.
 */

#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <algorithm>
#include <cstdint>
#include <cstring>

#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

using namespace std;

// FNV-1a Hash function for fast string fingerprinting
inline uint64_t fnv1a_hash(string_view s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (char c : s) {
        h ^= (uint8_t)c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Item struct optimized for size and comparison speed
// Aligned to 8 bytes for efficient access
struct alignas(8) Item {
    uint64_t hash;           // 8 bytes
    const char* str_ptr;     // 8 bytes
    uint32_t original_index; // 4 bytes
    uint16_t str_len;        // 2 bytes
    // 2 bytes padding to reach 24 bytes

    // Comparison optimized: hash -> length -> memcmp
    // This avoids expensive string comparisons unless strictly necessary (collisions)
    bool operator<(const Item& other) const {
        if (hash != other.hash) return hash < other.hash;
        if (str_len != other.str_len) return str_len < other.str_len;
        return memcmp(str_ptr, other.str_ptr, str_len) < 0;
    }

    bool operator!=(const Item& other) const {
        return hash != other.hash || str_len != other.str_len || memcmp(str_ptr, other.str_ptr, str_len) != 0;
    }
};

// Disjoint Set Union with parity tracking
// rel[i]: 0 if parallel to parent, 1 if perpendicular
struct DSU {
    vector<int> parent;
    vector<int> rel; 

    DSU(int n) {
        parent.resize(n);
        rel.assign(n, 0);
        for(int i = 0; i < n; ++i) parent[i] = i;
    }

    // Find with path compression
    pair<int, int> find(int i) {
        if (parent[i] != i) {
            pair<int, int> root = find(parent[i]);
            parent[i] = root.first;
            rel[i] ^= root.second;
        }
        return {parent[i], rel[i]};
    }

    // Unite sets. Returns false if contradiction.
    bool unite(int i, int j, int type) {
        pair<int, int> root_i = find(i);
        pair<int, int> root_j = find(j);
        
        int ri = root_i.first;
        int rj = root_j.first;
        // Needed relationship: rel(i, j) == type
        // rel(i, j) = rel(i, ri) ^ rel(ri, rj) ^ rel(j, rj)
        // type = rel(i) ^ rel(ri, rj) ^ rel(j)
        int needed = type ^ root_i.second ^ root_j.second;
        
        if (ri != rj) {
            parent[ri] = rj;
            rel[ri] = needed;
            return true;
        } else {
            // Check consistency
            return (needed == 0);
        }
    }

    // Flatten for O(1) read-only access
    void flatten() {
        for(int i = 0; i < (int)parent.size(); ++i) find(i);
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    // Read entire stdin into a large buffer
    vector<char> buffer;
    buffer.reserve(1 << 24); // 16MB initial
    
    while (cin) {
        size_t old_size = buffer.size();
        if (old_size == buffer.capacity()) {
            buffer.reserve(max<size_t>(old_size * 2, 1 << 24));
        }
        buffer.resize(buffer.capacity());
        cin.read(buffer.data() + old_size, buffer.size() - old_size);
        buffer.resize(old_size + cin.gcount());
    }

    if (buffer.empty()) return 0;

    const char* ptr = buffer.data();
    const char* end = ptr + buffer.size();

    // Helper lambdas for fast parsing
    auto skip_ws = [&]() {
        while (ptr < end && *ptr <= 32) ptr++;
    };

    auto read_token = [&]() -> string_view {
        skip_ws();
        if (ptr == end) return {};
        const char* start = ptr;
        while (ptr < end && *ptr > 32) ptr++;
        return string_view(start, ptr - start);
    };

    auto parse_int = [&](string_view s) {
        int res = 0;
        for (char c : s) res = res * 10 + (c - '0');
        return res;
    };

    string_view m_sv = read_token();
    string_view n_sv = read_token();
    if (m_sv.empty()) return 0;

    int m = parse_int(m_sv);
    int n = parse_int(n_sv);

    size_t total_items = 2ULL * (m + n);
    vector<Item> items(total_items);
    vector<uint8_t> obs_types(m);

    size_t idx = 0;
    // Parse Observations
    for(int i = 0; i < m; ++i) {
        string_view u = read_token();
        items[idx] = {fnv1a_hash(u), u.data(), (uint32_t)idx, (uint16_t)u.size()};
        idx++;
        string_view v = read_token();
        items[idx] = {fnv1a_hash(v), v.data(), (uint32_t)idx, (uint16_t)v.size()};
        idx++;
        string_view t = read_token();
        obs_types[i] = (t[0] == 'i' ? 1 : 0);
    }

    // Parse Queries
    for(int i = 0; i < n; ++i) {
        string_view u = read_token();
        items[idx] = {fnv1a_hash(u), u.data(), (uint32_t)idx, (uint16_t)u.size()};
        idx++;
        string_view v = read_token();
        items[idx] = {fnv1a_hash(v), v.data(), (uint32_t)idx, (uint16_t)v.size()};
        idx++;
    }

    // Parallel Sort to group identical strings
    parlay::sort_inplace(items);

    // Assign unique IDs
    vector<int> mapped_ids(total_items);
    int unique_count = 0;
    
    if (total_items > 0) {
        mapped_ids[items[0].original_index] = 0;
        for(size_t i = 1; i < total_items; ++i) {
            if (items[i] != items[i-1]) {
                unique_count++;
            }
            mapped_ids[items[i].original_index] = unique_count;
        }
        unique_count++;
    }

    // Build Graph (Sequential due to dependency)
    DSU dsu(unique_count);
    for(int i = 0; i < m; ++i) {
        if (!dsu.unite(mapped_ids[2*i], mapped_ids[2*i+1], obs_types[i])) {
            cout << "Waterloo\n";
            return 0;
        }
    }

    // Flatten DSU for lock-free parallel queries
    dsu.flatten();

    // Process Queries in Parallel
    parlay::sequence<uint8_t> results(n);
    size_t query_offset = 2ULL * m;

    parlay::parallel_for(0, n, [&](size_t i) {
        int u = mapped_ids[query_offset + 2*i];
        int v = mapped_ids[query_offset + 2*i + 1];

        int root_u = dsu.parent[u];
        int root_v = dsu.parent[v];

        if (root_u != root_v) {
            results[i] = 2; // unknown
        } else {
            results[i] = (dsu.rel[u] ^ dsu.rel[v]);
        }
    });

    // Generate Output in Parallel
    // Calculate lengths of output strings
    auto lengths = parlay::delayed_seq<int>(n, [&](size_t i) {
        return (results[i] == 0) ? 9 : (results[i] == 1 ? 10 : 8);
    });
    
    // Compute write offsets
    auto [offsets, total_size] = parlay::scan(lengths);
    
    vector<char> out_buf(total_size);
    
    // Parallel copy into output buffer
    parlay::parallel_for(0, n, [&](size_t i) {
        const char* s;
        int l;
        if (results[i] == 0) { s = "parallel\n"; l = 9; }
        else if (results[i] == 1) { s = "intersect\n"; l = 10; }
        else { s = "unknown\n"; l = 8; }
        
        memcpy(out_buf.data() + offsets[i], s, l);
    });
    
    cout.write(out_buf.data(), total_size);

    return 0;
}