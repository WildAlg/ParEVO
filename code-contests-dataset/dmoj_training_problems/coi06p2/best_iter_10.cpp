/*
 * Solution for Police System
 * 
 * Problem Analysis:
 * We need to determine if connectivity between two cities A and B is maintained under two conditions:
 * 1. An edge (road) is removed.
 * 2. A vertex (city) is removed.
 * 
 * Algorithm:
 * This problem can be efficiently solved using the Block-Cut Tree (BCT) data structure.
 * - The BCT represents the biconnected components (blocks) and articulation points of the graph.
 * - In the BCT, both original vertices and blocks are nodes. An edge exists between a vertex v and a block B
 *   if v is part of the component B.
 * - The BCT is a tree, allowing us to reduce connectivity queries to path queries.
 * 
 * Query Handling:
 * 1. Edge Cut (G1, G2):
 *    - An edge is a bridge if and only if its corresponding block in the BCT contains only that single edge.
 *    - Removing a bridge disconnects A and B iff the bridge's block node lies on the simple path between A and B in the BCT.
 *    - If the edge is not a bridge (part of a cycle), removing it never disconnects the graph.
 * 2. Vertex Cut (C):
 *    - Removing a vertex C disconnects A and B iff the vertex node C lies on the simple path between A and B in the BCT.
 * 
 * Implementation:
 * - Use DFS to find biconnected components and build the BCT.
 * - Preprocess the BCT for Lowest Common Ancestor (LCA) queries using binary lifting.
 * - Use LCA to check if a node is on the path between two other nodes in O(1) or O(log N).
 * - Parallelize sorting of adjacency lists and query processing using the Parlay library.
 * 
 * Complexity:
 * - Time: O((N + E + Q) * log N)
 * - Space: O(N + E)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
#include <cstring>
#include <cstdio>

// Parlay library for parallelism
#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Fast I/O Class
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
    
    inline void write_str(const char* s) {
        while (*s) putchar(*s++);
    }
} io;

// Constants
const int MAXN = 100005;
const int MAXE = 500005;
const int MAX_BCT = 200010; // Vertices + Blocks (approx 2*N)
const int LOGN = 19;

struct Edge {
    int v;
    int id;
    // Operator for sorting and binary search
    bool operator<(const Edge& other) const {
        return v < other.v;
    }
};

// Graph Data
vector<Edge> adj[MAXN];
pair<int, int> edges_list[MAXE];

// Block-Cut Tree Data
vector<int> bct_adj[MAX_BCT];
int edge_to_block[MAXE]; // Maps edge ID to Block ID
bool is_bridge_block[MAX_BCT]; // True if block consists of a single edge
int num_blocks = 0;

// DFS Variables for BCC
int tin[MAXN], low[MAXN], timer;
stack<int> st; // Stores edge IDs
int visited_token[MAXN]; // To handle duplicate vertices in block construction

// LCA Variables
int up[MAX_BCT][LOGN];
int tin_bct[MAX_BCT], tout_bct[MAX_BCT], timer_bct;

// Build Block-Cut Tree using DFS
void dfs_bcc(int u, int p = -1) {
    tin[u] = low[u] = ++timer;
    for (const auto& e : adj[u]) {
        int v = e.v;
        int id = e.id;
        if (v == p) continue;
        
        if (tin[v]) {
            // Back edge
            low[u] = min(low[u], tin[v]);
            if (tin[v] < tin[u]) {
                st.push(id);
            }
        } else {
            // Tree edge
            st.push(id);
            dfs_bcc(v, u);
            low[u] = min(low[u], low[v]);
            
            if (low[v] >= tin[u]) {
                // Found a biconnected component (block)
                int blk = MAXN + (++num_blocks);
                int edge_cnt = 0;
                
                // Add articulation point u to the block
                if (visited_token[u] != blk) {
                    bct_adj[u].push_back(blk);
                    bct_adj[blk].push_back(u);
                    visited_token[u] = blk;
                }
                
                // Pop edges from stack
                while (true) {
                    int eid = st.top();
                    st.pop();
                    edge_to_block[eid] = blk;
                    edge_cnt++;
                    
                    int x = edges_list[eid].first;
                    int y = edges_list[eid].second;
                    
                    // Add endpoints to the block in BCT
                    if (visited_token[x] != blk) {
                        bct_adj[x].push_back(blk);
                        bct_adj[blk].push_back(x);
                        visited_token[x] = blk;
                    }
                    if (visited_token[y] != blk) {
                        bct_adj[y].push_back(blk);
                        bct_adj[blk].push_back(y);
                        visited_token[y] = blk;
                    }
                    
                    if (eid == id) break;
                }
                
                // A block with 1 edge is a bridge
                is_bridge_block[blk] = (edge_cnt == 1);
            }
        }
    }
}

// Preprocess LCA on BCT
void dfs_lca(int u, int p) {
    tin_bct[u] = ++timer_bct;
    up[u][0] = p;
    for (int i = 1; i < LOGN; ++i) {
        up[u][i] = up[up[u][i-1]][i-1];
    }
    for (int v : bct_adj[u]) {
        if (v != p) dfs_lca(v, u);
    }
    tout_bct[u] = ++timer_bct;
}

// Check if u is an ancestor of v
inline bool is_ancestor(int u, int v) {
    return tin_bct[u] <= tin_bct[v] && tout_bct[u] >= tout_bct[v];
}

// Get Lowest Common Ancestor
inline int get_lca(int u, int v) {
    if (is_ancestor(u, v)) return u;
    if (is_ancestor(v, u)) return v;
    for (int i = LOGN - 1; i >= 0; --i) {
        if (!is_ancestor(up[u][i], v)) {
            u = up[u][i];
        }
    }
    return up[u][0];
}

// Check if 'target' lies on the simple path between u and v
inline bool on_path(int u, int v, int target) {
    int lca = get_lca(u, v);
    // target is on path if it is a descendant of LCA AND an ancestor of u or v
    return is_ancestor(lca, target) && (is_ancestor(target, u) || is_ancestor(target, v));
}

struct Query {
    int type;
    int A, B, C, G1, G2;
};

// Output buffer
char answers[300005][4];

int main() {
    int N, E;
    io.read(N);
    io.read(E);
    
    for (int i = 0; i < E; ++i) {
        int u, v;
        io.read(u);
        io.read(v);
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
        edges_list[i] = {u, v};
    }
    
    // Sort adjacency lists in parallel to allow binary search for edge lookup
    parlay::parallel_for(1, N + 1, [&](size_t i) {
        sort(adj[i].begin(), adj[i].end());
    });
    
    // Build BCT
    dfs_bcc(1);
    
    // Preprocess LCA
    dfs_lca(1, 1);
    
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
    
    // Process queries in parallel
    parlay::parallel_for(0, Q, [&](size_t i) {
        const auto& q = queries[i];
        bool possible = true;
        
        if (q.type == 1) {
            // Find edge ID for (G1, G2)
            const vector<Edge>& neighbors = adj[q.G1];
            auto it = lower_bound(neighbors.begin(), neighbors.end(), Edge{q.G2, 0});
            int eid = it->id;
            int blk = edge_to_block[eid];
            
            // If the block is a bridge, check if it lies on the path
            if (is_bridge_block[blk]) {
                if (on_path(q.A, q.B, blk)) {
                    possible = false;
                }
            }
        } else {
            // Check if vertex C lies on the path
            if (on_path(q.A, q.B, q.C)) {
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