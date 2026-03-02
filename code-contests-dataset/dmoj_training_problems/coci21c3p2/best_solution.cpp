/**
 * Problem: Maja's Christmas Tree
 * 
 * Solution Approach:
 * 1. Input Parsing: Use a custom `FastScanner` for high-performance I/O.
 * 2. Graph Construction:
 *    - Use a Compressed Sparse Row (CSR) representation.
 *    - CSR uses two flat arrays (`xadj`, `adj`) which is memory-efficient and cache-friendly.
 * 3. Component Identification:
 *    - Use a Disjoint Set Union (DSU) with union-by-size.
 *    - This allows us to group connected nodes and track component sizes on the fly.
 *    - We filter out small components (size 1 and 2) immediately as their diameters are trivial.
 *      This reduces the overhead for the parallel runtime.
 * 4. Parallel Diameter Calculation:
 *    - Use `parlay::map` to process "heavy" components (size >= 3) in parallel.
 *    - For each component, we compute the diameter using the Double BFS algorithm.
 *      - BFS 1: Find farthest node u from an arbitrary start node.
 *      - BFS 2: Find farthest node v from u. Distance(u, v) is the diameter.
 *    - Optimization: Use `thread_local` vectors for the BFS queue to avoid repeated memory allocations.
 *      We reserve capacity based on the component size to prevent reallocations during traversal.
 * 5. Aggregation: Sum the diameters using `parlay::reduce` and add the pre-calculated trivial diameters.
 * 
 * Complexity: O(N + M) time and space.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <numeric>

#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

// Fast Buffered Scanner for high-performance Input
class FastScanner {
    static const int BUFFER_SIZE = 1 << 18; // 256KB buffer
    char buffer[BUFFER_SIZE];
    int ptr, bytes_read;
    FILE *file;

    char read_char() {
        if (ptr == bytes_read) {
            ptr = 0;
            bytes_read = fread(buffer, 1, BUFFER_SIZE, file);
            if (bytes_read == 0) return EOF;
        }
        return buffer[ptr++];
    }

public:
    FastScanner(FILE *f = stdin) : file(f), ptr(0), bytes_read(0) {}

    template<typename T>
    bool scan(T &x) {
        int c = read_char();
        while (c != EOF && (c < '0' || c > '9')) c = read_char();
        if (c == EOF) return false;
        x = 0;
        while (c >= '0' && c <= '9') {
            x = x * 10 + (c - '0');
            c = read_char();
        }
        return true;
    }
};

// Disjoint Set Union (DSU) with Union by Size
// Optimized to use a single vector
struct DSU {
    vector<int> data; // negative: -size, non-negative: parent
    
    DSU(int n) {
        data.assign(n + 1, -1);
    }
    
    int find(int i) {
        int root = i;
        while (data[root] >= 0) root = data[root];
        int curr = i;
        while (curr != root) {
            int next = data[curr];
            data[curr] = root;
            curr = next;
        }
        return root;
    }
    
    void unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            // Union by size: attach smaller to larger
            if (data[root_i] > data[root_j]) swap(root_i, root_j);
            data[root_i] += data[root_j];
            data[root_j] = root_i;
        }
    }
    
    bool is_root(int i) const {
        return data[i] < 0;
    }
    
    int size(int i) const {
        return -data[i];
    }
};

// BFS State Node - optimized for queue
struct BFSNode {
    int u; // Current node
    int p; // Parent node
    int d; // Distance
};

struct Component {
    int root;
    int size;
};

int main() {
    // Disable standard I/O synchronization for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    FastScanner scanner;
    int n;
    long long m_in;
    if (!scanner.scan(n)) return 0;
    scanner.scan(m_in);
    int m = (int)m_in;

    struct Edge { int u, v; };
    vector<Edge> edges(m);
    
    // Reuse xadj for degree counting initially
    vector<int> xadj(n + 2, 0);
    DSU dsu(n);

    // Step 1: Read edges, build DSU, count degrees
    for (int i = 0; i < m; ++i) {
        int u, v;
        scanner.scan(u);
        scanner.scan(v);
        edges[i] = {u, v};
        xadj[u]++;
        xadj[v]++;
        dsu.unite(u, v);
    }

    // Step 2: Convert degrees to CSR offsets
    int current_offset = 0;
    for (int i = 1; i <= n; ++i) {
        int deg = xadj[i];
        xadj[i] = current_offset;
        current_offset += deg;
    }
    xadj[n + 1] = current_offset;
    
    // Step 3: Fill CSR adjacency array
    vector<int> adj(2 * m);
    vector<int> pos = xadj; // Copy offsets to track insertion positions

    for (const auto& e : edges) {
        adj[pos[e.u]++] = e.v;
        adj[pos[e.v]++] = e.u;
    }
    
    // Free temporary memory
    vector<Edge>().swap(edges);
    vector<int>().swap(pos);

    // Step 4: Identify components and filter trivial ones
    vector<Component> heavy_components;
    heavy_components.reserve(n);
    long long total_diameter = 0;

    for (int i = 1; i <= n; ++i) {
        if (dsu.is_root(i)) {
            int sz = dsu.size(i);
            if (sz == 1) {
                total_diameter += 1;
            } else if (sz == 2) {
                total_diameter += 2;
            } else {
                heavy_components.push_back({i, sz});
            }
        }
    }

    // Step 5: Parallel Diameter Calculation
    auto diameters = parlay::map(heavy_components, [&](const Component& comp) -> long long {
        // Thread-local BFS queue to avoid repeated allocations
        static thread_local vector<BFSNode> q;
        
        // Ensure capacity for the current component
        if (q.capacity() < (size_t)comp.size) {
            q.reserve(comp.size);
        }
        
        // Helper lambda: BFS to find the farthest node and its distance
        auto bfs = [&](int start_node) -> pair<int, int> {
            q.clear();
            q.push_back({start_node, -1, 0});
            
            int max_d = 0;
            int farthest = start_node;
            int head = 0;
            
            while(head < (int)q.size()) {
                BFSNode curr = q[head++];
                if (curr.d > max_d) {
                    max_d = curr.d;
                    farthest = curr.u;
                }
                
                int start = xadj[curr.u];
                int end = xadj[curr.u+1];
                for (int k = start; k < end; ++k) {
                    int v = adj[k];
                    // Tree traversal: avoid backtracking to parent
                    if (v != curr.p) {
                        q.push_back({v, curr.u, curr.d + 1});
                    }
                }
            }
            return {farthest, max_d};
        };

        // Double BFS Algorithm
        // 1. BFS from root to find farthest node u
        pair<int, int> p1 = bfs(comp.root);
        // 2. BFS from u to find farthest node v (distance is diameter)
        pair<int, int> p2 = bfs(p1.first);
        
        // Diameter in nodes = edges + 1
        return p2.second + 1;
    });

    // Step 6: Aggregate Results
    total_diameter += parlay::reduce(diameters);
    cout << total_diameter << endl;

    return 0;
}