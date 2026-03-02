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
    
    vector<string> grid(n);
    for (int i = 0; i < n; i++) {
        cin >> grid[i];
    }
    
    // Visited array to track processed cells
    vector<vector<bool>> visited(n, vector<bool>(m, false));
    
    // Counter for number of traps needed
    int trap_count = 0;
    
    // For each cell in the grid, perform BFS to trace the path
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            // If cell hasn't been visited yet
            if (!visited[i][j]) {
                // Track path to detect cycles
                vector<pair<int, int>> path;
                queue<pair<int, int>> q;
                
                // Start with current cell
                q.push({i, j});
                path.push_back({i, j});
                visited[i][j] = true;
                
                bool found_cycle = false;
                bool escaped = false;
                
                while (!q.empty() && !escaped && !found_cycle) {
                    auto [x, y] = q.front();
                    q.pop();
                    
                    char dir = grid[x][y];
                    int nx = x, ny = y;
                    
                    // Calculate next position based on direction
                    switch (dir) {
                        case 'N': nx--; break;
                        case 'E': ny++; break;
                        case 'S': nx++; break;
                        case 'W': ny--; break;
                    }
                    
                    // Check if we're out of bounds
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m) {
                        // We exit the grid, so no trap needed for this path
                        escaped = true;
                        break;
                    }
                    
                    // Check if we've already visited this cell in current path
                    bool already_in_path = false;
                    for (const auto& p : path) {
                        if (p.first == nx && p.second == ny) {
                            already_in_path = true;
                            break;
                        }
                    }
                    
                    // If we hit an already visited cell that was part of our current path, we found a cycle
                    if (already_in_path) {
                        found_cycle = true;
                        break;
                    }
                    
                    // Check if we've already visited this cell in any path
                    if (visited[nx][ny]) {
                        // If already visited in a different path, then we can safely skip
                        escaped = true;
                        break;
                    }
                    
                    // Add to visited and path
                    visited[nx][ny] = true;
                    path.push_back({nx, ny});
                    q.push({nx, ny});
                }
                
                // If we found a cycle (loop) within boundaries
                if (found_cycle && !escaped) {
                    trap_count++;
                }
            }
        }
    }
    
    cout << trap_count << endl;
    return 0;
}