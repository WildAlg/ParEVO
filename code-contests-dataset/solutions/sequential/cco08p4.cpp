#include <iostream>
#include <vector>
#include <string>

using namespace std;

// Disjoint Set Union (DSU) structure to manage components
struct DSU {
    vector<int> parent;

    // Initialize DSU with n elements, each in its own set
    DSU(int n) {
        parent.resize(n);
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    // Find the representative (root) of the set containing i
    // Uses path compression for efficiency
    int find(int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]);
    }

    // Unite the sets containing i and j
    // Returns true if they were in different sets, false otherwise
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = root_j;
            return true;
        }
        return false;
    }
};

int main() {
    // Fast I/O for performance
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<string> grid(n);
    for (int i = 0; i < n; i++) {
        cin >> grid[i];
    }

    // Total number of cells
    int total_cells = n * m;
    
    // Initialize DSU: Start assuming every cell is its own isolated component
    DSU dsu(total_cells);
    int components = total_cells;

    for (int r = 0; r < n; r++) {
        for (int c = 0; c < m; c++) {
            // Calculate current cell's 1D index
            int current_id = r * m + c;
            
            // Determine the coordinates of the next cell based on direction
            int next_r = r;
            int next_c = c;
            char dir = grid[r][c];

            if (dir == 'N') next_r--;
            else if (dir == 'S') next_r++;
            else if (dir == 'E') next_c++;
            else if (dir == 'W') next_c--;

            // Calculate next cell's 1D index
            int next_id = next_r * m + next_c;

            // Connect the current cell to the next cell
            // If they were previously in different components, we decrease the component count
            if (dsu.unite(current_id, next_id)) {
                components--;
            }
        }
    }

    // The answer is the number of disjoint sets remaining
    cout << components << "\n";

    return 0;
}