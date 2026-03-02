#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <array>
#include <algorithm>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/monoid.h>

// Mappings for characters J, O, I to integers 0, 1, 2 for modular arithmetic.
// This mapping allows the crossing operation to be represented as
// arithmetic modulo 3: cross(c1, c2) = -(c1 + c2) mod 3.
int char_to_int(char c) {
    if (c == 'J') return 0;
    if (c == 'O') return 1;
    return 2; // 'I'
}

// Custom modulo 3 function to handle negative results correctly.
// C++'s % operator can yield negative results for negative inputs.
int mod3(long long x) {
    return (x % 3 + 3) % 3;
}

// Node for the segment tree. Stores mismatch counts for each of the 9 possible
// reachable patterns and a lazy tag for range updates.
struct Node {
    std::array<int, 9> mismatches;
    int lazy_char;

    Node() : lazy_char(-1) { mismatches.fill(0); }
};

int N;
// The set of all reachable sequences forms an affine subspace over Z_3.
// This space can be described as S_A_base + span(U_base, W_base).
parlay::sequence<int> S_A_base, U_base, W_base;

// Type aliases for clarity.
using CountsPerPattern = std::array<int, 3>;
using AllCounts = std::array<CountsPerPattern, 9>;

// prefix_counts_AoS[i] stores the counts of {J, O, I} for each of the 9 patterns
// in the prefix of length i. This Array-of-Structs layout is for cache efficiency.
parlay::sequence<AllCounts> prefix_counts_AoS;

// The segment tree.
parlay::sequence<Node> tree;

// Threshold for switching from parallel to sequential recursion to avoid overhead.
constexpr int PARALLEL_THRESHOLD = 2048;

// Combine results from children nodes (2*u, 2*u+1) into the parent node (u).
void pull(int u) {
    for (int i = 0; i < 9; ++i) {
        tree[u].mismatches[i] = tree[2 * u].mismatches[i] + tree[2 * u + 1].mismatches[i];
    }
}

// Apply a lazy update to a node u, setting its range [tl, tr] to character c.
void apply_lazy(int u, int c, int tl, int tr) {
    tree[u].lazy_char = c;
    int len = tr - tl + 1;
    for (int k = 0; k < 9; ++k) {
        // Mismatches = length of range - number of matches.
        // Number of matches is found efficiently using precomputed prefix sums.
        int matches = prefix_counts_AoS[tr + 1][k][c] - prefix_counts_AoS[tl][k][c];
        tree[u].mismatches[k] = len - matches;
    }
}

// Push a lazy tag from a parent node u down to its children.
void push(int u, int tl, int tr) {
    if (tree[u].lazy_char != -1 && tl != tr) {
        int c = tree[u].lazy_char;
        int tm = tl + (tr - tl) / 2;
        apply_lazy(2 * u, c, tl, tm);
        apply_lazy(2 * u + 1, c, tm + 1, tr);
        tree[u].lazy_char = -1; // Clear lazy tag after pushing.
    }
}

// Build the segment tree from the initial target sequence T.
void build(const parlay::sequence<int>& T, int u, int tl, int tr) {
    tree[u].lazy_char = -1;
    if (tl == tr) {
        // Leaf node: calculate mismatches for a single position against all 9 patterns.
        for (int k = 0; k < 9; ++k) {
            int alpha = k / 3;
            int beta = k % 3;
            // Calculate G_k[tl] on the fly to save memory.
            int g_val = mod3((long long)S_A_base[tl] + alpha * U_base[tl] + beta * W_base[tl]);
            tree[u].mismatches[k] = (T[tl] != g_val);
        }
    } else {
        int tm = tl + (tr - tl) / 2;
        // Recurse on children. Use parallel recursion for large subproblems.
        if (tr - tl > PARALLEL_THRESHOLD) {
            parlay::par_do(
                [&] { build(T, 2 * u, tl, tm); },
                [&] { build(T, 2 * u + 1, tm + 1, tr); }
            );
        } else {
            build(T, 2 * u, tl, tm);
            build(T, 2 * u + 1, tm + 1, tr);
        }
        pull(u); // Combine results after children are built.
    }
}

// Update a range [l, r] in the segment tree to character c.
void update(int u, int tl, int tr, int l, int r, int c) {
    if (l > r) return;
    if (l == tl && r == tr) {
        // Node's range is fully contained in the update range. Apply lazy update.
        apply_lazy(u, c, tl, tr);
        return;
    }
    // Push down lazy tag before recursing.
    push(u, tl, tr);
    int tm = tl + (tr - tl) / 2;
    // Recurse to children that overlap with the update range.
    update(2 * u, tl, tm, l, std::min(r, tm), c);
    update(2 * u + 1, tm + 1, tr, std::max(l, tm + 1), r, c);
    pull(u); // Combine results after children are updated.
}

void solve() {
    std::cin >> N;
    std::string sa_str, sb_str, sc_str;
    std::cin >> sa_str >> sb_str >> sc_str;

    auto S_A_tmp = parlay::map(sa_str, char_to_int);
    auto S_B_tmp = parlay::map(sb_str, char_to_int);
    auto S_C_tmp = parlay::map(sc_str, char_to_int);
    
    S_A_base = std::move(S_A_tmp);
    // U_base and W_base are the basis vectors for the vector space part of the affine space.
    U_base = parlay::tabulate(N, [&](size_t i) { return mod3((long long)S_B_tmp[i] - S_A_base[i]); });
    W_base = parlay::tabulate(N, [&](size_t i) { return mod3((long long)S_C_tmp[i] - S_A_base[i]); });

    // Precompute prefix sums for all 9 patterns simultaneously using a single parallel scan.
    // This is more efficient than 9 separate scans due to better cache usage and less overhead.
    auto get_all_counts_at_pos = [&](size_t j) -> AllCounts {
        AllCounts counts_at_j;
        long long sa_j = S_A_base[j];
        long long u_j = U_base[j];
        long long w_j = W_base[j];
        for (int k = 0; k < 9; ++k) {
            int alpha = k / 3;
            int beta = k % 3;
            int val = mod3(sa_j + alpha * u_j + beta * w_j);
            if (val == 0) counts_at_j[k] = {1, 0, 0};
            else if (val == 1) counts_at_j[k] = {0, 1, 0};
            else counts_at_j[k] = {0, 0, 1};
        }
        return counts_at_j;
    };
    auto all_initial_counts = parlay::tabulate(N, get_all_counts_at_pos);

    auto add_all_counts = [](const AllCounts& a, const AllCounts& b) {
        AllCounts res;
        for (int k = 0; k < 9; ++k) {
            res[k][0] = a[k][0] + b[k][0];
            res[k][1] = a[k][1] + b[k][1];
            res[k][2] = a[k][2] + b[k][2];
        }
        return res;
    };
    
    AllCounts identity_all{}; // Zero-initialized

    auto [sums, total] = parlay::scan(all_initial_counts, parlay::make_monoid(add_all_counts, identity_all));
    prefix_counts_AoS = std::move(sums);
    prefix_counts_AoS.push_back(total);

    int Q;
    std::cin >> Q;
    std::string t0_str;
    std::cin >> t0_str;
    auto T = parlay::map(t0_str, char_to_int);

    tree.resize(4 * N + 4);
    build(T, 1, 0, N - 1);

    // Process T_0 and all Q subsequent updates.
    for (int j = 0; j <= Q; ++j) {
        bool possible = false;
        // A sequence is possible if it matches any of the 9 reachable patterns.
        // This is true if the total mismatch count at the root is 0 for any pattern.
        for (int k = 0; k < 9; ++k) {
            if (tree[1].mismatches[k] == 0) {
                possible = true;
                break;
            }
        }
        std::cout << (possible ? "Yes\n" : "No\n");

        if (j < Q) {
            int L, R;
            char C_char;
            std::cin >> L >> R >> C_char;
            int C = char_to_int(C_char);
            // Update the segment tree for the next candidate sequence.
            // Convert 1-based problem indexing to 0-based.
            update(1, 0, N - 1, L - 1, R - 1, C);
        }
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    solve();
    return 0;
}