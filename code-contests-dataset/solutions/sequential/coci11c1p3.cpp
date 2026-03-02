#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    // We need to count set bits for positions 0 to 19.
    // 2^20 = 1,048,576, which covers the input constraint (< 1,000,000).
    vector<int> bit_counts(20, 0);

    // Read input and count bits
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        for (int j = 0; j < 20; j++) {
            // Check if the j-th bit is set
            if ((x >> j) & 1) {
                bit_counts[j]++;
            }
        }
    }

    long long total_friendship_value = 0;

    // Calculate contribution of each bit position
    for (int j = 0; j < 20; j++) {
        long long ones = bit_counts[j];
        long long zeros = n - ones;

        // The number of pairs with different bits at this position is (ones * zeros).
        // Each such pair adds 2^j to the total sum.
        // We use 1LL to ensure the calculation is done in 64-bit integer arithmetic.
        long long contribution = (ones * zeros) * (1LL << j);
        
        total_friendship_value += contribution;
    }

    cout << total_friendship_value << "\n";

    return 0;
}