/**
 * JOI Open Contest - Digital Lines / Synchronization
 * 
 * Solution:
 * - Uses the "Synchronization" technique on a rooted tree.
 * - Tracks connected components using the "Component Root" concept.
 * - Components are maintained by tracking inactive edges on the path to the global root (node 1).
 * - A BIT is used to count inactive edges on paths.
 * - Binary Lifting is used to find the component root (highest ancestor in the same component).
 * - Parallelism is used for initialization and queries.
 * - Optimized graph storage (CSR) and iterative DFS.
 * 
 * Complexity: O((M + Q) log^2 N) time, O(N log N) space.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

using namespace std;

// ---------------------- Fast I/O ----------------------
class FastIO {
    static const int BUFFER_SIZE = 1 << 18;
    char buffer[BUFFER_SIZE];
    int p = BUFFER_SIZE;
    char out_buffer[BUFFER_SIZE];
    int out_p = 0;

    inline void load_buffer() {
        cin.read(buffer, BUFFER_SIZE);
        p = 0;
    }

    inline char read_char() {
        if (p == BUFFER_SIZE) load_buffer();
        return buffer[p++];
    }

public:
    inline int read_int() {
        int x = 0;
        char c = read_char();
        while (c < '0' || c > '9') {
            if (cin.eof()) return 0;
            c = read_char();
        }
        while (c >= '0' && c <= '9') {
            x = x * 10 + (c - '0');
            c = read_char();
        }
        return x;
    }

    inline void flush() {
        if (out_p > 0) {
            cout.write(out_buffer, out_p);
            out_p = 0;
        }
    }

    inline void write_int(int x) {
        if (out_p + 12 > BUFFER_SIZE) flush();
        if (x == 0) {
            out_buffer[out_p++] = '0';
            out_buffer[out_p++] = '\n';
            return;
        }
        char temp[12];
        int t_p = 0;
        while (x > 0) {
            temp[t_p++] = (x % 10) + '0';
            x /= 10;
        }
        while (t_p > 0) out_buffer[out_p++] = temp[--t_p];
        out_buffer[out_p++] = '\n';
    }

    ~FastIO() { flush(); }
} io;

// ---------------------- Constants & Globals ----------------------
const int MAXN = 100005;
const int LOGN = 17;

// Graph (CSR)
int head[MAXN];
struct Edge { int to, next, id; } edges[2 * MAXN];
int ecnt = 0;

inline void add_edge(int u, int v, int id) {
    ecnt++;
    edges[ecnt] = {v, head[u], id};
    head[u] = ecnt;
}

// Tree Properties (DFS ID)
int to_dfs[MAXN];       // Orig ID -> DFS ID
int tout[MAXN];         // DFS ID -> Subtree End
int parent_dfs[MAXN];   // DFS ID -> Parent DFS ID
int up[MAXN][LOGN];     // DFS ID -> Ancestor DFS ID
int edge_child[MAXN];   // Edge ID -> Child DFS ID
int timer = 0;

// BIT (DFS ID)
int bit[MAXN];
int N_val;

// State (DFS ID)
int val[MAXN];
int last_val[MAXN];
bool is_inactive[MAXN]; // Indexed by DFS ID (status of edge to parent)

// ---------------------- BIT Operations ----------------------
inline void bit_add(int i, int delta) {
    for (; i <= N_val; i += i & -i) bit[i] += delta;
}

inline void bit_range_add(int l, int r, int delta) {
    bit_add(l, delta);
    bit_add(r + 1, -delta);
}

inline int bit_query(int i) {
    int sum = 0;
    for (; i > 0; i -= i & -i) sum += bit[i];
    return sum;
}

// ---------------------- Tree Building ----------------------
void build_tree(int n) {
    // Iterative DFS
    static int st_u[MAXN];
    static int st_e[MAXN];
    int top = 0;

    st_u[0] = 1;
    st_e[0] = head[1];
    
    timer = 0;
    timer++;
    to_dfs[1] = 1;
    parent_dfs[1] = 0;
    
    while (top >= 0) {
        int u = st_u[top];
        int &e_idx = st_e[top];
        
        bool pushed = false;
        while (e_idx != 0) {
            int v = edges[e_idx].to;
            int id = edges[e_idx].id;
            e_idx = edges[e_idx].next;
            
            if (v != 1 && to_dfs[v] == 0) {
                timer++;
                int u_dfs = to_dfs[u];
                int v_dfs = timer;
                
                to_dfs[v] = v_dfs;
                parent_dfs[v_dfs] = u_dfs;
                edge_child[id] = v_dfs;
                
                // Binary Lifting
                up[v_dfs][0] = u_dfs;
                for (int i = 1; i < LOGN; i++) {
                    up[v_dfs][i] = up[up[v_dfs][i-1]][i-1];
                }
                
                top++;
                st_u[top] = v;
                st_e[top] = head[v];
                pushed = true;
                break;
            }
        }
        
        if (!pushed) {
            int u_dfs = to_dfs[u];
            tout[u_dfs] = timer;
            top--;
        }
    }
}

// ---------------------- Component Logic ----------------------
inline int find_root(int u) {
    if (u == 1) return 1;
    if (is_inactive[u]) return u; // Edge to parent is inactive

    int target = bit_query(u);
    if (target == 0) return 1; // Connected to global root
    
    int curr = u;
    for (int i = LOGN - 1; i >= 0; i--) {
        int anc = up[curr][i];
        if (anc != 0 && bit_query(anc) == target) {
            curr = anc;
        }
    }
    return curr;
}

// ---------------------- Main ----------------------
int main() {
    ios_base::sync_with_stdio(false);
    
    int N = io.read_int();
    int M = io.read_int();
    int Q = io.read_int();
    
    N_val = N;
    
    for (int i = 1; i < N; i++) {
        int u = io.read_int();
        int v = io.read_int();
        add_edge(u, v, i);
        add_edge(v, u, i);
    }
    
    vector<int> events(M);
    for (int i = 0; i < M; i++) events[i] = io.read_int();
    
    vector<int> queries(Q);
    for (int i = 0; i < Q; i++) queries[i] = io.read_int();
    
    build_tree(N);
    
    // Initial State: All lines are NOT built (inactive).
    // is_inactive[u] means edge (parent[u], u) is inactive.
    // For u=2..N, all are inactive.
    // BIT should reflect this: inactive edges contribute 1.
    for (int i = 2; i <= N; i++) {
        bit_range_add(i, tout[i], 1);
    }
    
    parlay::parallel_for(1, N + 1, [&](int i) {
        val[i] = 1;
        last_val[i] = 0;
        if (i > 1) is_inactive[i] = true;
    });
    
    for (int t = 0; t < M; t++) {
        int e_idx = events[t];
        int u = edge_child[e_idx]; // Child node in DFS ID
        
        if (is_inactive[u]) {
            // Build Line: Inactive -> Active
            bit_range_add(u, tout[u], -1);
            is_inactive[u] = false;
            
            // Merge u into p
            int p = parent_dfs[u];
            int root_p = find_root(p);
            
            val[root_p] += val[u] - last_val[u];
            
        } else {
            // Remove Line: Active -> Inactive
            bit_range_add(u, tout[u], 1);
            is_inactive[u] = true;
            
            // Split u from p
            int p = parent_dfs[u];
            int root_p = find_root(p);
            
            val[u] = val[root_p];
            last_val[u] = val[root_p];
        }
    }
    
    vector<int> results(Q);
    parlay::parallel_for(0, Q, [&](int i) {
        int u_orig = queries[i];
        int u_dfs = to_dfs[u_orig];
        results[i] = val[find_root(u_dfs)];
    });
    
    for (int i = 0; i < Q; i++) {
        io.write_int(results[i]);
    }
    
    return 0;
}