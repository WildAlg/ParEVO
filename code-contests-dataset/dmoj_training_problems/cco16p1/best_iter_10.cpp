/*
 * Canadian Computing Olympiad: 2016 Day 1, Problem 1 - Field Trip
 * Solution using DSU and Parallel Reduction via Parlay
 * 
 * Strategy:
 * 1. Efficient Input Parsing: Uses a buffered FastScanner for reading large input.
 * 2. Component Identification: Uses Disjoint Set Union (DSU) to group students into connected components.
 *    Tracks component size and edge count to distinguish between Paths and Cycles.
 * 3. Parallel Processing: Uses Parlay's delayed_seq and reduce to calculate the optimal students/cuts
 *    for each component in parallel and aggregate the results.
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/delayed.h>

using namespace std;

// Fast Input Scanner for efficient IO with large datasets
class FastScanner {
    static const int BUF_SIZE = 1 << 18;
    char buf[BUF_SIZE];
    int ptr, len;
    FILE* in;

    char next_char() {
        if (ptr == len) {
            ptr = 0;
            len = fread(buf, 1, BUF_SIZE, in);
            if (len == 0) return EOF;
        }
        return buf[ptr++];
    }

public:
    FastScanner() : ptr(0), len(0), in(stdin) {}

    template<typename T>
    bool read(T& x) {
        x = 0;
        char c = next_char();
        while (c != EOF && c <= 32) c = next_char();
        if (c == EOF) return false;
        // Input integers are non-negative
        while (c >= '0' && c <= '9') {
            x = x * 10 + (c - '0');
            c = next_char();
        }
        return true;
    }
};

// Disjoint Set Union (DSU) structure to manage connected components
struct DSU {
    vector<int> parent;
    vector<int> sz;
    vector<int> edges;

    DSU(int n) {
        parent.resize(n + 1);
        iota(parent.begin(), parent.end(), 0);
        sz.assign(n + 1, 1);
        edges.assign(n + 1, 0);
    }

    // Iterative find with path compression
    int find(int i) {
        int root = i;
        while (root != parent[root]) root = parent[root];
        
        int curr = i;
        while (curr != root) {
            int next = parent[curr];
            parent[curr] = root;
            curr = next;
        }
        return root;
    }

    // Union by size
    void unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            if (sz[root_i] < sz[root_j]) swap(root_i, root_j);
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
    FastScanner sc;
    
    int N, M;
    long long K;
    
    if (!sc.read(N)) return 0;
    sc.read(M);
    sc.read(K);

    DSU dsu(N);

    for(int i = 0; i < M; ++i) {
        int u, v;
        sc.read(u);
        sc.read(v);
        dsu.unite(u, v);
    }

    // Define the processing logic for each student index
    // This lambda will be executed in parallel
    auto process = [&](size_t idx) -> Result {
        int r = (int)idx + 1;
        
        // We only process the root of each component to avoid double counting
        // If parent[r] != r, it's not a root
        if (dsu.parent[r] != r) return {0, 0};

        long long L = dsu.sz[r];
        if (L < K) return {0, 0};

        long long E = dsu.edges[r];
        long long p = L / K;
        long long s = p * K;
        long long c = 0;
        long long rem = L % K;

        // Components are either Cycles (Edges == Vertices) or Paths (Edges == Vertices - 1)
        if (E == L) { // Cycle
            if (L == K) {
                c = 0; // Perfect cycle, take all, no cuts needed
            } else if (rem == 0) {
                c = p; // Cut p edges to get p segments of size K
            } else {
                c = p + 1; // Cut p+1 edges to isolate p segments
            }
        } else { // Path
            if (rem == 0) {
                c = p - 1; // p-1 internal cuts to separate p segments
            } else {
                c = p; // p cuts: internal cuts + cut off remainder
            }
        }
        return {s, c};
    };

    // Create a delayed sequence representing the results for students 1..N
    // Delayed sequence avoids storing the intermediate Result objects
    auto results = parlay::delayed_seq<Result>(N, process);

    // Monoid for summing results
    auto monoid = parlay::make_monoid([](Result a, Result b) {
        return Result{a.students + b.students, a.cuts + b.cuts};
    }, Result{0, 0});

    // Parallel reduction to compute total students and cuts
    Result total = parlay::reduce(results, monoid);

    cout << total.students << " " << total.cuts << "\n";

    return 0;
}