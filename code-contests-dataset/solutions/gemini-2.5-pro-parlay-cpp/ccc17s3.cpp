#include <iostream>
#include <vector>
#include <algorithm>
#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/monoid.h>

int main() {
    // Fast I/O is crucial for competitive programming problems with large inputs.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int N;
    std::cin >> N;

    // The maximum possible length of a single piece of wood is 2000, per problem spec.
    const int MAX_L_SPEC = 2000;

    // Step 1: Count frequencies of each wood length and find the max length present.
    // A sequential loop is highly efficient for this task. It avoids parallel overhead
    // and is typically I/O-bound anyway. We also find the maximum length actually
    // present in the input to optimize the search space for heights.
    std::vector<int> counts(MAX_L_SPEC + 1, 0);
    int max_l_present = 0;
    for (int i = 0; i < N; ++i) {
        int l;
        std::cin >> l;
        counts[l]++;
        if (l > max_l_present) {
            max_l_present = l;
        }
    }

    // A board is formed by two pieces. Minimum possible board height: 1 + 1 = 2.
    const int MIN_H = 2;
    // Optimization: Maximum possible board height is twice the longest piece actually available.
    // This can significantly reduce the number of heights to check.
    const int MAX_H = 2 * max_l_present;

    // With N>=2 and L_i>=1, max_l_present>=1, so MAX_H>=2. The range of heights to check
    // will not be empty, so no special empty-case handling is needed.

    // Step 2: Compute the number of boards for each possible height in parallel.
    // parlay::tabulate is a clean and efficient parallel pattern for generating a
    // sequence where each element is a function of its index.
    auto fence_lengths_per_height = parlay::tabulate(MAX_H - MIN_H + 1, [&](size_t i) {
        int H = i + MIN_H; // Map the index `i` to the actual height `H`.
        int current_fence_length = 0;
        
        // To form a board of height H, we need two pieces l1 and l2 where l1 + l2 = H.
        // We iterate through the smaller piece length l1; l2 is then determined.
        // Looping up to H/2 avoids double-counting pairs (l1, l2) and (l2, l1).
        
        // Optimization: since l2 = H - l1 and l2 must be <= MAX_L_SPEC,
        // it implies that l1 must be >= H - MAX_L_SPEC.
        int start_l1 = std::max(1, H - MAX_L_SPEC);

        for (int l1 = start_l1; l1 <= H / 2; ++l1) {
            int l2 = H - l1;
            
            if (l1 < l2) {
                // If pieces have different lengths, the number of boards is
                // limited by the piece with the smaller count.
                current_fence_length += std::min(counts[l1], counts[l2]);
            } else { // This case is l1 == l2, which happens only when H is even.
                // If pieces have the same length, we can form count/2 pairs.
                current_fence_length += counts[l1] / 2;
            }
        }
        return current_fence_length;
    });

    // Step 3: Find the maximum fence length and count how many heights achieve it.
    // This two-pass approach (reduce for max, then count) is robust and performant.
    
    // Find the maximum fence length. Since N>=2, at least one board can be made,
    // so the max length will be at least 1.
    int max_len = parlay::reduce(fence_lengths_per_height, parlay::maxm<int>());

    // Count how many heights achieve this maximum length.
    long long num_heights_with_max_len = parlay::count(fence_lengths_per_height, max_len);

    // Step 4: Output the final result as required.
    std::cout << max_len << " " << num_heights_with_max_len << std::endl;

    return 0;
}