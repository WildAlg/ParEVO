#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <parlay/sequence.h>
#include <parlay/primitives.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    vector<int> lengths(n);
    for (int i = 0; i < n; i++) {
        cin >> lengths[i];
    }
    
    // Count frequency of each length
    map<int, int> freq;
    for (int len : lengths) {
        freq[len]++;
    }
    
    // Calculate all possible board lengths and their counts
    map<int, int> board_counts;
    
    // For each unique length, calculate how many boards we can make
    for (auto& pair : freq) {
        int len = pair.first;
        int count = pair.second;
        
        // If we have at least 2 pieces of this length, we can make boards
        if (count >= 2) {
            board_counts[len + len] += count / 2;
        }
        
        // For other lengths, we need to pair with different lengths
        // Only consider pairs where we have at least one of each
        for (auto& other_pair : freq) {
            int other_len = other_pair.first;
            int other_count = other_pair.second;
            
            if (len < other_len && other_count > 0) {
                // We can make boards of length len + other_len
                board_counts[len + other_len] += min(count, other_count);
            }
        }
    }
    
    // Find maximum fence length and count of different heights
    int max_length = 0;
    int height_count = 0;
    
    for (auto& pair : board_counts) {
        int board_length = pair.first;
        int num_boards = pair.second;
        
        if (num_boards > max_length) {
            max_length = num_boards;
            height_count = 1;
        } else if (num_boards == max_length) {
            height_count++;
        }
    }
    
    cout << max_length << " " << height_count << endl;
    
    return 0;
}