#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

int main() {
    int n, m;
    cin >> n >> m;
    
    // Read the grid
    vector<string> grid(n);
    for (int i = 0; i < n; i++) {
        cin >> grid[i];
    }
    
    // Create a visited matrix to track which cells have been processed
    vector<vector<bool>> visited(n, vector<bool>(m, false));
    
    // For each cell, determine if it leads to a cycle or path that ends outside the grid
    int traps_needed = 0;
    
    // Process each cell
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (visited[i][j]) continue;
            
            // Start a path from this cell
            vector<pair<int, int>> path;
            int x = i, y = j;
            bool in_cycle = false;
            
            // Follow the path until we either exit the grid or detect a cycle
            while (x >= 0 && x < n && y >= 0 && y < m) {
                if (visited[x][y]) {
                    // We've seen this cell before - check if it's part of a cycle
                    // Find where this cell appeared in our current path
                    auto it = find(path.begin(), path.end(), make_pair(x, y));
                    if (it != path.end()) {
                        in_cycle = true;
                        break;
                    } else {
                        break; // Already processed, exit
                    }
                }
                
                path.push_back({x, y});
                visited[x][y] = true;
                
                // Move according to the direction
                char dir = grid[x][y];
                if (dir == 'N') x--;
                else if (dir == 'S') x++;
                else if (dir == 'E') y++;
                else if (dir == 'W') y--;
            }
            
            // If we exited the grid, this path doesn't need a trap
            // If we found a cycle, we need a trap at the start of the cycle
            if (!in_cycle) {
                // Path leads out of the grid - no trap needed
                continue;
            } else {
                // This cycle needs a trap
                traps_needed++;
            }
        }
    }
    
    cout << traps_needed << endl;
    return 0;
}