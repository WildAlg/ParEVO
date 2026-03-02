/**
 * Problem: Pizza Delivery
 * Approach:
 * 1. The city grid allows horizontal movement anywhere, but vertical movement is restricted to columns 1 and C.
 * 2. This structure suggests a "highway" system formed by the first and last columns.
 * 3. We construct a graph (ladder graph) representing these two columns and the horizontal connections between them.
 *    - Nodes: (r, 1) and (r, C) for all rows r.
 *    - Edges: Vertical moves in col 1 and C, and horizontal traversal of row r.
 * 4. We precompute All-Pairs Shortest Paths (APSP) on this ladder graph using Dijkstra's algorithm.
 *    - Since the graph is small (2*R vertices), we run Dijkstra from each node in parallel.
 * 5. For each delivery step (from current location to next), the shortest path is either:
 *    - Direct horizontal move (if in same row).
 *    - Moving to a portal (col 1 or C), traveling via the ladder graph to the destination row's portal, and moving to the destination.
 * 6. We process all delivery legs in parallel and sum the costs.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

const long long INF = 1e18;

struct Edge {
    int to;
    int weight;
};

int R, C;
vector<vector<int>> grid;
vector<vector<long long>> row_prefix;
vector<vector<Edge>> adj;

// Run Dijkstra from a single source node on the ladder graph
void run_dijkstra(int start_node, vector<long long>& d) {
    int n = 2 * R;
    d.assign(n, INF);
    d[start_node] = 0;
    
    // Min-priority queue
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;
    pq.push({0, start_node});
    
    while (!pq.empty()) {
        long long d_u = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        
        if (d_u > d[u]) continue;
        
        for (const auto& edge : adj[u]) {
            if (d[u] + edge.weight < d[edge.to]) {
                d[edge.to] = d[u] + edge.weight;
                pq.push({d[edge.to], edge.to});
            }
        }
    }
}

// Calculate distance between two cells in the same row
// Path excludes start cell (c1) and includes end cell (c2)
long long get_row_dist(int r, int c1, int c2) {
    if (c1 == c2) return 0;
    if (c1 < c2) {
        // Sum of weights from c1+1 to c2
        return row_prefix[r][c2] - row_prefix[r][c1];
    } else {
        // Sum of weights from c2 to c1-1
        // P[r][c1-1] - P[r][c2-1]
        long long val1 = row_prefix[r][c1-1];
        long long val2 = (c2 == 0) ? 0 : row_prefix[r][c2-1];
        return val1 - val2;
    }
}

int main() {
    // Optimize standard I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> R >> C)) return 0;

    grid.resize(R, vector<int>(C));
    row_prefix.resize(R, vector<long long>(C));

    // Read grid and build prefix sums for each row
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            cin >> grid[i][j];
            if (j == 0) row_prefix[i][j] = grid[i][j];
            else row_prefix[i][j] = row_prefix[i][j-1] + grid[i][j];
        }
    }

    // Build the ladder graph
    // Vertices 0 to R-1 represent (r, 0) [Left Column]
    // Vertices R to 2R-1 represent (r, C-1) [Right Column]
    int num_nodes = 2 * R;
    adj.resize(num_nodes);

    for (int r = 0; r < R; r++) {
        int u_left = r;
        int u_right = r + R;

        // Vertical edges in Left Column (Col 1)
        if (r > 0) adj[u_left].push_back({r - 1, grid[r-1][0]});
        if (r < R - 1) adj[u_left].push_back({r + 1, grid[r+1][0]});

        // Vertical edges in Right Column (Col C)
        if (r > 0) adj[u_right].push_back({r - 1 + R, grid[r-1][C-1]});
        if (r < R - 1) adj[u_right].push_back({r + 1 + R, grid[r+1][C-1]});

        // Horizontal edges connecting Left and Right columns of row r
        // Left -> Right: cost is sum of weights from col 1 to C-1
        long long cost_lr = 0;
        if (C > 1) cost_lr = row_prefix[r][C-1] - row_prefix[r][0];
        adj[u_left].push_back({u_right, (int)cost_lr});

        // Right -> Left: cost is sum of weights from col C-2 to 0
        long long cost_rl = 0;
        if (C > 1) cost_rl = row_prefix[r][C-2]; 
        adj[u_right].push_back({u_left, (int)cost_rl});
    }

    // Precompute APSP on the ladder graph in parallel
    vector<vector<long long>> portal_dists(num_nodes);
    parlay::parallel_for(0, num_nodes, [&](size_t i) {
        run_dijkstra(i, portal_dists[i]);
    });

    int D;
    if (!(cin >> D)) return 0;

    // Store locations: Start at (0,0) then D deliveries
    parlay::sequence<pair<int, int>> locs(D + 1);
    locs[0] = {0, 0}; // 0-based indexing for (1, 1)
    for(int i = 0; i < D; ++i) {
        cin >> locs[i+1].first >> locs[i+1].second;
        locs[i+1].first--; // Convert to 0-based
        locs[i+1].second--;
    }

    // Calculate cost for each segment in parallel
    auto costs = parlay::tabulate(D, [&](size_t i) {
        int r1 = locs[i].first;
        int c1 = locs[i].second;
        int r2 = locs[i+1].first;
        int c2 = locs[i+1].second;

        long long min_dist = INF;

        // Option 1: Direct horizontal move (only valid if same row)
        if (r1 == r2) {
            min_dist = min(min_dist, get_row_dist(r1, c1, c2));
        }

        // Option 2: Via portals (Left or Right column)
        // Costs to reach portals in row r1
        long long cost_to_PL = get_row_dist(r1, c1, 0);   // To (r1, 0)
        long long cost_to_PR = get_row_dist(r1, c1, C-1); // To (r1, C-1)

        // Costs from portals in row r2 to destination
        long long cost_from_DL = get_row_dist(r2, 0, c2);   // From (r2, 0)
        long long cost_from_DR = get_row_dist(r2, C-1, c2); // From (r2, C-1)

        // Check all 4 combinations of entry/exit portals
        // Path: Start -> Portal1 -> ... -> Portal2 -> End
        // Note: portal_dists includes weight of Portal2 but excludes Portal1
        // get_row_dist includes weight of destination but excludes source
        // This chains correctly.

        // Via Left Portal -> Left Portal
        min_dist = min(min_dist, cost_to_PL + portal_dists[r1][r2] + cost_from_DL);
        // Via Left Portal -> Right Portal
        min_dist = min(min_dist, cost_to_PL + portal_dists[r1][r2 + R] + cost_from_DR);
        // Via Right Portal -> Left Portal
        min_dist = min(min_dist, cost_to_PR + portal_dists[r1 + R][r2] + cost_from_DL);
        // Via Right Portal -> Right Portal
        min_dist = min(min_dist, cost_to_PR + portal_dists[r1 + R][r2 + R] + cost_from_DR);

        return min_dist;
    });

    // Total time is start node weight + sum of segment costs
    long long total_time = grid[0][0] + parlay::reduce(costs);
    cout << total_time << endl;

    return 0;
}