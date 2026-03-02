#include <iostream>
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
    
    long long result = 0;
    
    // For each bit position, count how many numbers have 0 and how many have 1
    // The contribution to the final sum is the product of these counts
    for (int bit = 0; bit < 30; bit++) { // 30 bits is enough for numbers < 1000000
        // Create a sequence indicating whether each number has bit set to 1
        auto bit_flags = parlay::map(numbers, [bit](long long x) {
            return (x >> bit) & 1;
        });
        
        // Count zeros and ones in parallel
        long long count_1 = parlay::reduce(bit_flags, parlay::plus<long long>());
        long long count_0 = N - count_1;
        
        // For this bit position, the total contribution is count_0 * count_1
        result += count_0 * count_1 * (1LL << bit);
    }
    
    cout << result << endl;
    return 0;
}