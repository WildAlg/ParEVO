/*
 * Canadian Computing Olympiad: 2016 Day 1, Problem 1 - Field Trip
 * Solution using Compact DSU and Parlay Parallel Reduction
 */
#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <cstdint>

// Parlay library includes
#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/delayed.h>

using namespace std;

// High-performance Input Scanner
class FastScanner {
    static const int BUF_SIZE = 1 << 18;
    char buf[BUF_SIZE];
    int ptr, len;
    FILE* in;

    // Inline next char for speed
    inline char next() {
        if (ptr == len) {
            ptr = 0;
            len = fread(buf, 1, BUF_SIZE, in);
            if (len == 0) return EOF;
        }
        return buf[ptr++];
    }

public:
    FastScanner() : ptr(0), len(0), in(stdin) {}
    
    // Optimized integer reading
    inline int readInt() {
        int x = 0;
        char c = next();
        // Skip whitespace
        while (c <= ' ') {
            if (c == EOF) return 0;
            c = next();
        }
        // Read digits
        while (c >= '0') {
            x = x * 10 + (c - '0');
            c = next();
        }
        return x;
    }
};

// Compact DSU structure
struct DSU {
    // p[i] < 0: i is root, -p[i] is size
    // p[i] >= 0: p[i] is parent
    vector<int> p;
    // Track if component is a cycle
    vector<uint8_t> is_cyc;

    DSU(int n) : p(n + 1, -1), is_cyc(n + 1, 0) {}

    // Non-recursive find with path compression
    inline int find(int i) {
        int root = i;
        while (p[root] >= 0) root = p[root];
        
        while (i != root) {
            int tmp = p[i];
            p[i] = root;
            i = tmp;
        }
        return root;
    }

    // Unite by size
    inline void unite(int i, int j) {
        int r1 = find(i);
        int r2 = find(j);
        if (r1 != r2) {
            // Merge smaller into larger
            if (p[r1] > p[r2]) swap(r1, r2);
            p[r1] += p[r2];
            p[r2] = r1;
            // Note: Since max degree is 2, we never merge a cycle with anything.
            // A cycle is closed and cannot be extended.
        } else {
            // Closing a cycle
            is_cyc[r1] = 1;
        }
    }
};

int main() {
    // Use FastScanner for input
    FastScanner sc;
    int N = sc.readInt();
    int M = sc.readInt();
    int K = sc.readInt();

    DSU dsu(N);
    for (int i = 0; i < M; ++i) {
        dsu.unite(sc.readInt(), sc.readInt());
    }

    // Parallel reduction using Parlay
    // delayed_seq creates a lazy view, processing elements on demand
    // We use long long to prevent any potential overflow, though int fits constraints
    auto results = parlay::delayed_seq<pair<long long, long long>>(N + 1, [&](size_t i) {
        // Only process component roots
        if (i == 0 || dsu.p[i] >= 0) return make_pair(0LL, 0LL);

        long long L = -dsu.p[i];
        if (L < K) return make_pair(0LL, 0LL);

        long long p = L / K;
        long long rem = L % K;
        long long s = p * K;
        long long c = 0;

        // Calculate cuts
        // For a path component:
        // We want p segments of size K.
        // If rem == 0: we just cut the path into p segments (p-1 cuts).
        // If rem > 0: we cut p segments and isolate them from remainder (p cuts).
        long long path_cuts = (rem == 0) ? p - 1 : p;

        if (dsu.is_cyc[i]) {
            // For a cycle component:
            // If L == K, we take the whole cycle (0 cuts).
            // Otherwise, we first cut 1 edge to make it a path, then apply path logic.
            // Result is path_cuts + 1.
            if (L == K) {
                c = 0;
            } else {
                c = path_cuts + 1;
            }
        } else {
            c = path_cuts;
        }
        return make_pair(s, c);
    });

    // Reduce results using summation monoid
    auto total = parlay::reduce(results, parlay::make_monoid([](pair<long long, long long> a, pair<long long, long long> b) {
        return make_pair(a.first + b.first, a.second + b.second);
    }, make_pair(0LL, 0LL)));

    // Output result
    printf("%lld %lld\n", total.first, total.second);

    return 0;
}