/*
 * Solution for "Little sheep Be" problem.
 * 
 * Algorithm Overview:
 * 1. Graph Construction: 
 *    - Uses Compressed Sparse Row (CSR) format with Structure of Arrays (SoA) for optimal memory layout.
 *    - Precomputes edge weights (distances) during graph construction.
 * 2. Component Decomposition: 
 *    - A sequential BFS linearizes the graph into `nodes_list`.
 *    - Nodes of each connected component are stored contiguously in BFS discovery order.
 *    - This topological property enables a fast linear-scan solver without internal queues.
 * 3. Parallel Solving (Parlay):
 *    - Each component is processed independently in parallel.
 *    - We model radii as r_u = K_u * x + C_u, where x is the root's radius.
 *    - Constraints r_u >= 0 define a valid interval [L, R] for x.
 *    - Cycles provide consistency checks (even cycles) or fix x (odd cycles).
 *    - If x is not fixed, we minimize sum(r_u^2) via quadratic optimization.
 * 4. Fast I/O:
 *    - Custom buffer-based input reading significantly reduces overhead compared to cin/scanf.
 * 
 * Complexity: O(N + M) time and space.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdio>
#include <cstring>

#include <parlay/parallel.h>
#include <parlay/sequence.h>

using namespace std;

// Types
using Real = double;
const Real EPS = 1e-7;
const Real INF = 1e18;

struct Point {
    Real x, y;
};

// Global Data
int n, m;
vector<Point> coords;

// Graph (CSR / SoA)
vector<int> row_ptr;
vector<int> col_idx;
vector<Real> weights;

// Component Decomposition
vector<int> nodes_list;
vector<pair<int, int>> comp_ranges;

// Solver State (SoA)
vector<int> K; // 0: unvisited, 1 or -1: coefficient
vector<Real> C; // constant
vector<Real> final_radii;

// Fast I/O
const int BUF_SIZE = 1 << 18;
char buffer[BUF_SIZE];
int buf_pos = 0, buf_len = 0;

inline char read_char() {
    if (buf_pos >= buf_len) {
        buf_pos = 0;
        buf_len = fread(buffer, 1, BUF_SIZE, stdin);
        if (buf_len == 0) return 0;
    }
    return buffer[buf_pos++];
}

inline int read_int() {
    int x = 0;
    char c = read_char();
    while (c <= ' ') {
        if (c == 0) return 0;
        c = read_char();
    }
    bool neg = false;
    if (c == '-') { neg = true; c = read_char(); }
    while (c >= '0' && c <= '9') {
        x = x * 10 + (c - '0');
        c = read_char();
    }
    return neg ? -x : x;
}

inline Real read_double() {
    Real x = 0.0;
    char c = read_char();
    while (c <= ' ') {
        if (c == 0) return 0.0;
        c = read_char();
    }
    bool neg = false;
    if (c == '-') { neg = true; c = read_char(); }
    while (c >= '0' && c <= '9') {
        x = x * 10.0 + (c - '0');
        c = read_char();
    }
    if (c == '.') {
        Real div = 1.0;
        c = read_char();
        while (c >= '0' && c <= '9') {
            x = x * 10.0 + (c - '0');
            div *= 10.0;
            c = read_char();
        }
        x /= div;
    }
    return neg ? -x : x;
}

// Build Graph
void build_graph(const vector<pair<int, int>>& edges) {
    vector<int> degree(n, 0);
    for (const auto& e : edges) {
        degree[e.first]++;
        degree[e.second]++;
    }

    row_ptr.resize(n + 1);
    row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        row_ptr[i+1] = row_ptr[i] + degree[i];
    }

    int m_double = row_ptr[n];
    col_idx.resize(m_double);
    weights.resize(m_double);

    vector<int> cur = row_ptr;
    for (const auto& e : edges) {
        int u = e.first;
        int v = e.second;

        Real dx = coords[u].x - coords[v].x;
        Real dy = coords[u].y - coords[v].y;
        Real w = std::sqrt(dx*dx + dy*dy);

        int pu = cur[u]++;
        int pv = cur[v]++;

        col_idx[pu] = v;
        weights[pu] = w;
        col_idx[pv] = u;
        weights[pv] = w;
    }
}

// Solver
bool solve_component(int start_idx, int count) {
    if (count == 0) return true;

    int root = nodes_list[start_idx];
    K[root] = 1;
    C[root] = 0.0;

    bool x_fixed = false;
    Real fixed_val = 0.0;

    Real L = -INF;
    Real R = INF;
    Real sum_kc = 0.0;

    for (int i = 0; i < count; ++i) {
        int u = nodes_list[start_idx + i];
        int Ku = K[u];
        Real Cu = C[u];

        // Constraint r_u >= 0
        if (Ku == 1) {
            // x >= -C
            if (-Cu > L) L = -Cu;
            sum_kc += Cu;
        } else {
            // x <= C
            if (Cu < R) R = Cu;
            sum_kc -= Cu;
        }

        // Neighbors
        int e_start = row_ptr[u];
        int e_end = row_ptr[u+1];

        for (int j = e_start; j < e_end; ++j) {
            int v = col_idx[j];
            Real w = weights[j];

            if (K[v] == 0) {
                // Unvisited in BFS tree
                K[v] = -Ku;
                C[v] = w - Cu;
            } else {
                // Visited
                // r_u + r_v = w
                // (Ku + Kv)x + (Cu + Cv) = w
                int k_sum = Ku + K[v];
                Real c_sum = Cu + C[v];

                if (k_sum == 0) {
                    // Even cycle
                    if (std::abs(c_sum - w) > EPS) return false;
                } else {
                    // Odd cycle, fixes x
                    // 2*Ku*x = w - c_sum (if Kv == Ku)
                    Real val = (w - c_sum) / k_sum;
                    if (x_fixed) {
                        if (std::abs(val - fixed_val) > EPS) return false;
                    } else {
                        x_fixed = true;
                        fixed_val = val;
                    }
                }
            }
        }
    }

    if (L > R + EPS) return false;

    Real x_final;
    if (x_fixed) {
        if (fixed_val < L - EPS || fixed_val > R + EPS) return false;
        x_final = fixed_val;
    } else {
        // Minimize sum (r_i)^2 => x = -sum(KC) / N
        Real x_opt = -sum_kc / count;
        if (x_opt < L) x_final = L;
        else if (x_opt > R) x_final = R;
        else x_final = x_opt;
    }

    // Set radii
    for (int i = 0; i < count; ++i) {
        int u = nodes_list[start_idx + i];
        final_radii[u] = K[u] * x_final + C[u];
    }
    return true;
}

int main() {
    // Input
    n = read_int();
    m = read_int();

    coords.resize(n);
    for (int i = 0; i < n; ++i) {
        coords[i].x = read_double();
        coords[i].y = read_double();
    }

    vector<pair<int, int>> edges(m);
    for (int i = 0; i < m; ++i) {
        edges[i].first = read_int() - 1;
        edges[i].second = read_int() - 1;
    }

    build_graph(edges);

    // Decomposition
    nodes_list.resize(n);
    vector<char> visited(n, 0);
    int list_tail = 0;

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            int start_idx = list_tail;
            visited[i] = 1;
            nodes_list[list_tail++] = i;

            int head = start_idx;
            while (head < list_tail) {
                int u = nodes_list[head++];
                int e_start = row_ptr[u];
                int e_end = row_ptr[u+1];
                for (int j = e_start; j < e_end; ++j) {
                    int v = col_idx[j];
                    if (!visited[v]) {
                        visited[v] = 1;
                        nodes_list[list_tail++] = v;
                    }
                }
            }
            comp_ranges.push_back({start_idx, list_tail - start_idx});
        }
    }

    // Prepare solver memory
    K.assign(n, 0);
    C.resize(n);
    final_radii.resize(n);

    // Parallel Solve
    parlay::sequence<bool> results(comp_ranges.size());
    parlay::parallel_for(0, comp_ranges.size(), [&](size_t i) {
        results[i] = solve_component(comp_ranges[i].first, comp_ranges[i].second);
    });

    for (bool r : results) {
        if (!r) {
            printf("NE\n");
            return 0;
        }
    }

    printf("DA\n");
    for (int i = 0; i < n; ++i) {
        printf("%.6f\n", final_radii[i]);
    }

    return 0;
}