#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Maximum length of a single piece of wood is 2000.
// Therefore, maximum height of a fence (sum of two pieces) is 4000.
const int MAX_LEN = 2000;
const int MAX_HEIGHT = 4000;

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    // Frequency array to store how many pieces of each length we have.
    // Size is 2001 to handle 1-based indexing up to 2000.
    vector<int> freq(MAX_LEN + 1, 0);

    for (int i = 0; i < n; i++) {
        int l;
        cin >> l;
        freq[l]++;
    }

    int max_fence_len = 0;
    int num_heights = 0;

    // Iterate through all possible fence heights (sum of two boards).
    // H can range from 1+1=2 to 2000+2000=4000.
    for (int h = 1; h <= MAX_HEIGHT; h++) {
        int current_len = 0;

        // Iterate through possible lengths for the first piece (A).
        // We go up to h/2 to avoid double counting pairs (A, B) and (B, A).
        for (int a = 1; a <= h / 2; a++) {
            int b = h - a;
            
            // Check if the required second piece length (B) is valid.
            if (b > MAX_LEN) continue; 
            
            if (a == b) {
                // Case: Both pieces are the same length.
                // We can form freq[a] / 2 boards.
                current_len += freq[a] / 2;
            } else {
                // Case: Pieces are different lengths.
                // The number of boards is limited by the piece with the smaller count.
                current_len += min(freq[a], freq[b]);
            }
        }

        // Update the global maximums
        if (current_len > max_fence_len) {
            // Found a new longest fence length
            max_fence_len = current_len;
            num_heights = 1; // Reset count of heights
        } else if (current_len == max_fence_len) {
            // Found another height that achieves the same max length
            num_heights++;
        }
    }

    cout << max_fence_len << " " << num_heights << "\n";

    return 0;
}