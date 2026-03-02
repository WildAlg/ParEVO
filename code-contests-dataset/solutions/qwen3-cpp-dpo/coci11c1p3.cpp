#include <iostream>
#include <vector>
#include <algorithm>
#include <parlay/sequence.h>
#include <parlay/primitives.h>

using namespace std;

int main() {
    int N;
    cin >> N;
    
    parlay::sequence<long long> numbers(N);
    for (int i = 0; i < N; i++) {
        cin >> numbers[i];
    }
    
    // For each bit position, count how many numbers have 0 and how many have 1
    long long result = 0;
    
    // Process bits from most significant to least significant
    // We'll handle up to 30 bits since numbers are less than 1,000,000
    for (int bit = 29; bit >= 0; bit--) {
        long long count_zero = 0;
        long long count_one = 0;
        
        // Count numbers with 0 and 1 at this bit position
        for (long long num : numbers) {
            if ((num >> bit) & 1) {
                count_one++;
            } else {
                count_zero++;
            }
        }
        
        // Calculate contribution to total friendship value
        // For each pair (i,j) where i != j, if their bits differ at this position,
        // they contribute 2^bit to the friendship value
        // Total pairs with different bits = count_zero * count_one
        result += count_zero * count_one * (1LL << bit);
    }
    
    cout << result << endl;
    
    return 0;
}