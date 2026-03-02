#include <iostream>
#include <vector>
#include <queue>
#include <string>

using namespace std;

// Structure to represent a coordinate in 3D space
struct Point {
    int x, y, z;
};

// Arrays to represent the 6 possible movement directions
// (Forward, Backward, Right, Left, Up, Down)
int dx[] = {1, -1, 0, 0, 0, 0};
int dy[] = {0, 0, 1, -1, 0, 0};
int dz[] = {0, 0, 0, 0, 1, -1};

// 3D arrays to store the grid and distances
// Max N is 100, so 105 is safe.
int grid[305][305][305]; // Indices: [z][x][y]
int dist[305][305][305];

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    int xs, ys, zs;
    cin >> xs >> ys >> zs;

    int xe, ye, ze;
    cin >> xe >> ye >> ze;

    // Initialize distances to -1
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                dist[k][i][j] = -1;
            }
        }
    }

    // Read the grid
    // Input format: N matrices.
    // i-th matrix is height i (z).
    // Rows are x, Cols are y.
    for (int z = 1; z <= n; z++) {
        for (int x = 1; x <= n; x++) {
            string row;
            cin >> row;
            for (int y = 1; y <= n; y++) {
                // Convert char '0'/'1' to int
                grid[z][x][y] = row[y - 1] - '0';
            }
        }
    }

    // BFS Queue
    queue<Point> q;

    // Start position setup
    q.push({xs, ys, zs});
    dist[zs][xs][ys] = 0;

    while (!q.empty()) {
        Point curr = q.front();
        q.pop();

        // Check if we reached the target
        if (curr.x == xe && curr.y == ye && curr.z == ze) {
            cout << dist[curr.z][curr.x][curr.y] << "\n";
            return 0;
        }

        // Try all 6 directions
        for (int i = 0; i < 6; i++) {
            int nx = curr.x + dx[i];
            int ny = curr.y + dy[i];
            int nz = curr.z + dz[i];

            // Check bounds (1 to N)
            if (nx >= 1 && nx <= n && ny >= 1 && ny <= n && nz >= 1 && nz <= n) {
                // Check if not a cloud (0) and not visited (-1)
                if (grid[nz][nx][ny] == 0 && dist[nz][nx][ny] == -1) {
                    dist[nz][nx][ny] = dist[curr.z][curr.x][curr.y] + 1;
                    q.push({nx, ny, nz});
                }
            }
        }
    }

    // If queue is empty and target not reached
    cout << "-1\n";

    return 0;
}