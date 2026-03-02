/**
 * Robots - Competitive Programming Solution
 * 
 * Algorithm:
 * 1. Model the problem as a shortest path query on a functional graph.
 * 2. Nodes are defined by (Field ID, Exit Direction). Edges represent moving to the next field and turning.
 *    Since movement is deterministic, each node has at most one outgoing edge (functional graph).
 * 3. Precompute graph components (cycles + trees) to answer distance queries in O(1).
 *    - Detect cycles and label nodes with component ID and position in cycle.
 *    - For tree parts, use DFS on the reversed graph to compute depth and root (cycle node).
 *    - Use entry/exit times (Euler tour) to check ancestor relationships in tree parts.
 * 4. For each query:
 *    - Check direct reachability (0 turns) using spatial calculations.
 *    - Determine entry nodes (state after 1st turn) and target nodes (state before last straight move).
 *    - Compute min distance using precomputed graph properties.
 * 
 * Complexity: O((K + Q) log K)
 * - Graph construction: O(K log K)
 * - Graph preprocessing: O(K)
 * - Query processing: O(Q log K)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <tuple>
#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

using namespace std;

// Data structures
struct Field {
    int r, c;
    char type; // 'L' or 'R'
    int id; // 0 to K-1
};

struct NodeData {
    int cycle_id;      // ID of the component's cycle (-1 if not reaching a cycle)
    int pos_in_cycle;  // Position in cycle (0 to len-1)
    int root;          // The cycle node (or dead-end node) this node reaches
    int dist_to_root;  // Distance to the root
    int tin, tout;     // DFS entry/exit times for tree ancestry
};

// Globals
int N, M, K, Q;
vector<Field> fields;
map<int, vector<pair<int, int>>> rows_map; // r -> sorted list of (c, field_id)
map<int, vector<pair<int, int>>> cols_map; // c -> sorted list of (r, field_id)

int num_nodes;
vector<int> adj;
vector<vector<int>> rev_adj;
vector<NodeData> nodes;
vector<int> cycle_lens; // Size of each cycle
int timer_dfs;

// Helper to find next field
// Returns field_id or -1
int get_next_field(int r, int c, int dir) {
    if (dir == 0) { // Up
        auto it = cols_map.find(c);
        if (it == cols_map.end()) return -1;
        const auto& vec = it->second;
        auto lit = lower_bound(vec.begin(), vec.end(), make_pair(r, -1));
        if (lit == vec.begin()) return vec.back().second;
        else return prev(lit)->second;
    } else if (dir == 2) { // Down
        auto it = cols_map.find(c);
        if (it == cols_map.end()) return -1;
        const auto& vec = it->second;
        auto lit = upper_bound(vec.begin(), vec.end(), make_pair(r, 2000000));
        if (lit == vec.end()) return vec.front().second;
        else return lit->second;
    } else if (dir == 1) { // Right
        auto it = rows_map.find(r);
        if (it == rows_map.end()) return -1;
        const auto& vec = it->second;
        auto lit = upper_bound(vec.begin(), vec.end(), make_pair(c, 2000000));
        if (lit == vec.end()) return vec.front().second;
        else return lit->second;
    } else { // Left
        auto it = rows_map.find(r);
        if (it == rows_map.end()) return -1;
        const auto& vec = it->second;
        auto lit = lower_bound(vec.begin(), vec.end(), make_pair(c, -1));
        if (lit == vec.begin()) return vec.back().second;
        else return prev(lit)->second;
    }
}

// Check direct reachability (0 turns)
bool can_reach_direct(int r1, int c1, int r2, int c2, int dir) {
    if (dir == 0 || dir == 2) { if (c1 != c2) return false; }
    if (dir == 1 || dir == 3) { if (r1 != r2) return false; }

    int next_f = get_next_field(r1, c1, dir);
    
    // If no field, reachable (infinite loop)
    if (next_f == -1) return true;

    int r_next = fields[next_f].r;
    int c_next = fields[next_f].c;

    long long dist_target = 0;
    long long dist_next = 0;
    
    if (dir == 0) { // Up
        dist_target = (long long)(r1 - r2 + N) % N;
        dist_next = (long long)(r1 - r_next + N) % N;
        if (dist_next == 0) dist_next = N;
    } else if (dir == 2) { // Down
        dist_target = (long long)(r2 - r1 + N) % N;
        dist_next = (long long)(r_next - r1 + N) % N;
        if (dist_next == 0) dist_next = N;
    } else if (dir == 1) { // Right
        dist_target = (long long)(c2 - c1 + M) % M;
        dist_next = (long long)(c_next - c1 + M) % M;
        if (dist_next == 0) dist_next = M;
    } else { // Left
        dist_target = (long long)(c1 - c2 + M) % M;
        dist_next = (long long)(c1 - c_next + M) % M;
        if (dist_next == 0) dist_next = M;
    }
    
    if (dist_target == 0) return true;
    return dist_target <= dist_next;
}

// DFS for tree processing
void dfs_tree(int u, int r, int d) {
    nodes[u].root = r;
    nodes[u].dist_to_root = d;
    nodes[u].tin = ++timer_dfs;
    for (int v : rev_adj[u]) {
        if (nodes[v].cycle_id == -1) { // Don't go back into cycle
            dfs_tree(v, r, d + 1);
        }
    }
    nodes[u].tout = ++timer_dfs;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N >> M >> K)) return 0;
    
    fields.resize(K);
    for (int i = 0; i < K; i++) {
        cin >> fields[i].r >> fields[i].c >> fields[i].type;
        fields[i].id = i;
        rows_map[fields[i].r].push_back({fields[i].c, i});
        cols_map[fields[i].c].push_back({fields[i].r, i});
    }

    // Sort fields for binary search
    for (auto& kv : rows_map) sort(kv.second.begin(), kv.second.end());
    for (auto& kv : cols_map) sort(kv.second.begin(), kv.second.end());

    num_nodes = 4 * K;
    adj.assign(num_nodes, -1);
    rev_adj.resize(num_nodes);

    // Build functional graph in parallel
    parlay::parallel_for(0, K, [&](int i) {
        for (int d = 0; d < 4; d++) {
            int u = 4 * i + d;
            int next_f = get_next_field(fields[i].r, fields[i].c, d);
            if (next_f != -1) {
                int new_d;
                if (fields[next_f].type == 'L') new_d = (d + 3) % 4;
                else new_d = (d + 1) % 4;
                adj[u] = 4 * next_f + new_d;
            }
        }
    });

    for (int u = 0; u < num_nodes; u++) {
        if (adj[u] != -1) rev_adj[adj[u]].push_back(u);
    }

    // Graph analysis (Components, Cycles, Trees)
    nodes.resize(num_nodes);
    for(int i=0; i<num_nodes; ++i) {
        nodes[i].cycle_id = -1;
        nodes[i].root = -1;
    }
    
    vector<int> visited(num_nodes, 0); // 0: new, 1: active, 2: visited
    int cycle_count = 0;

    for (int i = 0; i < num_nodes; i++) {
        if (visited[i] == 2) continue;
        int curr = i;
        vector<int> path;
        while (curr != -1 && visited[curr] == 0) {
            visited[curr] = 1;
            path.push_back(curr);
            curr = adj[curr];
        }

        if (curr != -1 && visited[curr] == 1) {
            // Cycle detected
            vector<int> cycle_nodes;
            bool in_cycle = false;
            for (int node : path) {
                if (node == curr) in_cycle = true;
                if (in_cycle) cycle_nodes.push_back(node);
            }
            
            int len = cycle_nodes.size();
            cycle_lens.push_back(len);
            for (int k = 0; k < len; k++) {
                int node = cycle_nodes[k];
                nodes[node].cycle_id = cycle_count;
                nodes[node].pos_in_cycle = k;
                visited[node] = 2;
            }
            cycle_count++;
        }
        for (int node : path) visited[node] = 2;
    }

    // Process trees
    timer_dfs = 0;
    // 1. Trees rooted at cycle nodes
    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].cycle_id != -1) {
            dfs_tree(i, i, 0);
        }
    }
    // 2. Trees rooted at dead ends (sinks)
    for (int i = 0; i < num_nodes; i++) {
        if (adj[i] == -1 && nodes[i].root == -1) {
             dfs_tree(i, i, 0);
        }
    }

    cin >> Q;
    vector<tuple<int, int, int, int>> queries(Q);
    for(int i=0; i<Q; ++i) {
        cin >> get<0>(queries[i]) >> get<1>(queries[i]) >> get<2>(queries[i]) >> get<3>(queries[i]);
    }

    vector<int> results(Q);

    // Process queries in parallel
    parlay::parallel_for(0, Q, [&](int i) {
        int r1, c1, r2, c2;
        tie(r1, c1, r2, c2) = queries[i];

        int ans = 2e9;

        // Step 1: Check 0 turns
        for (int d = 0; d < 4; d++) {
            if (can_reach_direct(r1, c1, r2, c2, d)) {
                ans = 0;
                break;
            }
        }

        if (ans != 0) {
            // Step 2: Use graph
            // Identify entry nodes
            vector<int> entries;
            for (int d = 0; d < 4; d++) {
                int f = get_next_field(r1, c1, d);
                if (f != -1) {
                    int new_d;
                    if (fields[f].type == 'L') new_d = (d + 3) % 4;
                    else new_d = (d + 1) % 4;
                    entries.push_back(4 * f + new_d);
                }
            }

            // Identify target nodes
            vector<int> targets;
            for (int d = 0; d < 4; d++) {
                int opp_d = (d + 2) % 4;
                int f = get_next_field(r2, c2, opp_d);
                if (f != -1) {
                    targets.push_back(4 * f + d);
                }
            }

            for (int u : entries) {
                for (int v : targets) {
                    int d_graph = -1;
                    
                    int root_u = nodes[u].root;
                    int root_v = nodes[v].root;
                    
                    if (root_u != -1) { // u must reach somewhere
                        if (nodes[v].cycle_id != -1) { 
                            // v is on a cycle
                            // u must reach the same cycle component
                            if (nodes[root_u].cycle_id == nodes[v].cycle_id) {
                                int len = cycle_lens[nodes[v].cycle_id];
                                int diff = (nodes[v].pos_in_cycle - nodes[root_u].pos_in_cycle + len) % len;
                                d_graph = nodes[u].dist_to_root + diff;
                            }
                        } else {
                            // v is in a tree (or dead end)
                            // u must be in v's subtree in reversed graph
                            if (root_u == root_v) {
                                if (nodes[v].tin <= nodes[u].tin && nodes[u].tout <= nodes[v].tout) {
                                    d_graph = nodes[u].dist_to_root - nodes[v].dist_to_root;
                                }
                            }
                        }
                    }

                    if (d_graph != -1) {
                        ans = min(ans, 1 + d_graph);
                    }
                }
            }
        }

        if (ans > 1e9) results[i] = -1;
        else results[i] = ans;
    });

    for (int x : results) cout << x << "\n";

    return 0;
}