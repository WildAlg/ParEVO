/*
 * Solution for Police System
 * 
 * Algorithm:
 * 1. Graph Representation: Compressed Sparse Row (CSR) for memory efficiency and cache locality.
 * 2. Biconnected Components: Tarjan's algorithm to build Block-Cut Tree (BCT).
 *    - Uses static arrays (linked list style) for BCT edges to avoid dynamic allocation overhead.
 * 3. LCA: Euler Tour + Sparse Table (RMQ) for O(1) LCA queries.
 * 4. Parallelism: Uses Parlay library for sorting adjacency lists and processing queries.
 * 5. Optimization:
 *    - Search for edges in the smaller adjacency list for Type 1 queries.
 *    - Static memory allocation for BCT graph.
 * 
 * Complexity:
 * - Time: O(N + E + Q) roughly, with O(1) per query.
 * - Space: O(N + E).
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>

#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Fast I/O
struct FastIO {
    static const int S = 1 << 20;
    char buf[S], *p1, *p2;
    char outbuf[S], *p_out;
    
    FastIO() : p1(buf), p2(buf), p_out(outbuf) {}
    ~FastIO() { flush(); }
    
    inline char getchar() {
        if (p1 == p2) {
            p2 = (p1 = buf) + fread(buf, 1, S, stdin);
            if (p1 == p2) return EOF;
        }
        return *p1++;
    }
    
    inline void flush() {
        if (p_out != outbuf) {
            fwrite(outbuf, 1, p_out - outbuf, stdout);
            p_out = outbuf;
        }
    }
    
    inline void putchar(char c) {
        if (p_out == outbuf + S) flush();
        *p_out++ = c;
    }
    
    inline void write_str(const char* s) {
        while (*s) putchar(*s++);
    }
    
    template <typename T>
    inline void read(T &x) {
        char c = getchar();
        x = 0;
        bool neg = false;
        while (c < '0' || c > '9') {
            if (c == '-') neg = true;
            c = getchar();
        }
        while (c >= '0' && c <= '9') {
            x = x * 10 + (c - '0');
            c = getchar();
        }
        if (neg) x = -x;
    }
} io;

const int MAXN = 100005;
const int MAXE = 500005;
const int MAX_BCT_NODES = 200020; // Vertices + Blocks
const int MAX_BCT_EDGES = 400040; // 2 * MAX_BCT_NODES
const int LOG_MAX_TOUR = 20;

struct EdgeInfo {
    int v;
    int id;
    bool operator<(const EdgeInfo& other) const {
        return v < other.v;
    }
};

// CSR Graph
int head[MAXN];
int degree[MAXN];
EdgeInfo adj_pool[2 * MAXE];
pair<int, int> raw_edges[MAXE];

// BCT Graph (Static Linked List)
int bct_head[MAX_BCT_NODES];
int bct_next[MAX_BCT_EDGES];
int bct_to[MAX_BCT_EDGES];
int bct_edge_cnt = 0;

inline void add_bct_edge(int u, int v) {
    bct_to[bct_edge_cnt] = v;
    bct_next[bct_edge_cnt] = bct_head[u];
    bct_head[u] = bct_edge_cnt++;
    
    bct_to[bct_edge_cnt] = u;
    bct_next[bct_edge_cnt] = bct_head[v];
    bct_head[v] = bct_edge_cnt++;
}

int edge_to_block[MAXE];
bool is_bridge_block[MAX_BCT_NODES];
int num_blocks = 0;

// BCC DFS
int tin[MAXN], low[MAXN], timer;
int stack_arr[MAXE];
int stack_top = 0;
int visited_token[MAXN]; 

// LCA
int tin_bct[MAX_BCT_NODES], tout_bct[MAX_BCT_NODES], timer_bct;
int depth[MAX_BCT_NODES];
int euler_tour[MAX_BCT_NODES * 2];
int first_occ[MAX_BCT_NODES];
int euler_timer = 0;
int st_min[LOG_MAX_TOUR][MAX_BCT_NODES * 2];

void dfs_bcc(int u, int p = -1) {
    tin[u] = low[u] = ++timer;
    for (int i = head[u]; i < head[u] + degree[u]; ++i) {
        int v = adj_pool[i].v;
        int id = adj_pool[i].id;
        if (v == p) continue;
        
        if (tin[v]) {
            low[u] = min(low[u], tin[v]);
            if (tin[v] < tin[u]) {
                stack_arr[stack_top++] = id;
            }
        } else {
            stack_arr[stack_top++] = id;
            dfs_bcc(v, u);
            low[u] = min(low[u], low[v]);
            
            if (low[v] >= tin[u]) {
                int blk = MAXN + (++num_blocks);
                int edge_cnt = 0;
                
                if (visited_token[u] != blk) {
                    add_bct_edge(u, blk);
                    visited_token[u] = blk;
                }
                
                while (true) {
                    int eid = stack_arr[--stack_top];
                    edge_to_block[eid] = blk;
                    edge_cnt++;
                    
                    int x = raw_edges[eid].first;
                    int y = raw_edges[eid].second;
                    
                    if (visited_token[x] != blk) {
                        add_bct_edge(x, blk);
                        visited_token[x] = blk;
                    }
                    if (visited_token[y] != blk) {
                        add_bct_edge(y, blk);
                        visited_token[y] = blk;
                    }
                    
                    if (eid == id) break;
                }
                is_bridge_block[blk] = (edge_cnt == 1);
            }
        }
    }
}

void dfs_lca(int u, int p, int d) {
    tin_bct[u] = ++timer_bct;
    depth[u] = d;
    first_occ[u] = euler_timer;
    euler_tour[euler_timer++] = u;
    
    for (int i = bct_head[u]; i != -1; i = bct_next[i]) {
        int v = bct_to[i];
        if (v != p) {
            dfs_lca(v, u, d + 1);
            euler_tour[euler_timer++] = u;
        }
    }
    tout_bct[u] = ++timer_bct;
}

void build_st() {
    parlay::parallel_for(0, euler_timer, [&](size_t i) {
        st_min[0][i] = euler_tour[i];
    });
    
    for (int j = 1; j < LOG_MAX_TOUR; j++) {
        int len = euler_timer - (1 << j) + 1;
        if (len <= 0) break;
        parlay::parallel_for(0, len, [&](size_t i) {
            int u = st_min[j-1][i];
            int v = st_min[j-1][i + (1 << (j-1))];
            st_min[j][i] = (depth[u] < depth[v]) ? u : v;
        });
    }
}

inline int get_lca(int u, int v) {
    int l = first_occ[u];
    int r = first_occ[v];
    if (l > r) { int temp = l; l = r; r = temp; }
    int k = 31 - __builtin_clz(r - l + 1);
    int u_node = st_min[k][l];
    int v_node = st_min[k][r - (1 << k) + 1];
    return (depth[u_node] < depth[v_node]) ? u_node : v_node;
}

inline bool is_ancestor(int u, int v) {
    return tin_bct[u] <= tin_bct[v] && tout_bct[u] >= tout_bct[v];
}

struct Query {
    int type;
    int A, B, C, G1, G2;
};

char answers[300005][4];

int main() {
    memset(bct_head, -1, sizeof(bct_head));
    
    int N, E;
    io.read(N);
    io.read(E);
    
    for (int i = 0; i < E; ++i) {
        int u, v;
        io.read(u);
        io.read(v);
        raw_edges[i] = {u, v};
        degree[u]++;
        degree[v]++;
    }
    
    head[1] = 0;
    for (int i = 1; i < N; ++i) {
        head[i+1] = head[i] + degree[i];
    }
    
    vector<int> cur_head(N + 1);
    for(int i=1; i<=N; ++i) cur_head[i] = head[i];
    
    for (int i = 0; i < E; ++i) {
        int u = raw_edges[i].first;
        int v = raw_edges[i].second;
        adj_pool[cur_head[u]++] = {v, i};
        adj_pool[cur_head[v]++] = {u, i};
    }
    
    parlay::parallel_for(1, N + 1, [&](size_t i) {
        std::sort(adj_pool + head[i], adj_pool + head[i] + degree[i]);
    });
    
    dfs_bcc(1);
    
    dfs_lca(1, 1, 0);
    build_st();
    
    int Q;
    io.read(Q);
    
    vector<Query> queries(Q);
    for (int i = 0; i < Q; ++i) {
        io.read(queries[i].type);
        if (queries[i].type == 1) {
            io.read(queries[i].A);
            io.read(queries[i].B);
            io.read(queries[i].G1);
            io.read(queries[i].G2);
        } else {
            io.read(queries[i].A);
            io.read(queries[i].B);
            io.read(queries[i].C);
        }
    }
    
    parlay::parallel_for(0, Q, [&](size_t i) {
        const auto& q = queries[i];
        bool possible = true;
        
        if (q.type == 1) {
            int u = q.G1;
            int v = q.G2;
            if (degree[u] > degree[v]) {
                int temp = u; u = v; v = temp;
            }
            
            int eid = -1;
            int l = 0, r = degree[u] - 1;
            EdgeInfo* base = adj_pool + head[u];
            
            while (l <= r) {
                int mid = l + (r - l) / 2;
                if (base[mid].v == v) {
                    eid = base[mid].id;
                    break;
                } else if (base[mid].v < v) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
            
            if (eid != -1) {
                int blk = edge_to_block[eid];
                if (is_bridge_block[blk]) {
                    int lca = get_lca(q.A, q.B);
                    if (is_ancestor(lca, blk) && (is_ancestor(blk, q.A) || is_ancestor(blk, q.B))) {
                        possible = false;
                    }
                }
            }
        } else {
            int C = q.C;
            int lca = get_lca(q.A, q.B);
            if (is_ancestor(lca, C) && (is_ancestor(C, q.A) || is_ancestor(C, q.B))) {
                possible = false;
            }
        }
        
        if (possible) {
            answers[i][0] = 'y'; answers[i][1] = 'e'; answers[i][2] = 's'; answers[i][3] = '\0';
        } else {
            answers[i][0] = 'n'; answers[i][1] = 'o'; answers[i][2] = '\0';
        }
    });
    
    for (int i = 0; i < Q; ++i) {
        io.write_str(answers[i]);
        io.putchar('\n');
    }
    
    return 0;
}