#include <iostream>
#include <vector>
#include <algorithm>

// Include necessary parlay headers for parallel algorithms and data structures.
// primitives.h provides core algorithms like tabulate and reduce.
// sequence.h provides the parlay::sequence container.
// monoid.h is needed for custom reduction operations.
// delayed_sequence.h provides view-based operations like delayed_map.
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/monoid.h>
#include <parlay/delayed_sequence.h>

// A structure to hold the maximum fence length and the count of heights achieving it.
// This allows finding both in a single parallel reduction pass, which is more
// efficient than two separate passes (e.g., reduce for max, then count).
struct MaxFenceInfo {
    int maxLength;
    long count;
};

// The combining function for our custom monoid used in the parallel reduction.
// It merges two MaxFenceInfo objects:
// - If one maxLength is greater, it is chosen as the new maximum.
// - If the maxLengths are equal, their counts are summed.
MaxFenceInfo combine_max_info(const MaxFenceInfo& a, const MaxFenceInfo& b) {
    if (a.maxLength > b.maxLength) {
        return a;
    }
    if (b.maxLength > a.maxLength) {
        return b;
    }
    // If maxLengths are equal, they represent the current max, so we combine their counts.
    return {a.maxLength, a.count + b.count};
}

int main() {
    // Fast I/O is standard practice in competitive programming for performance.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int N;
    std::cin >> N;

    // Use constexpr for compile-time constants for clarity and potential optimization.
    constexpr int MAX_WOOD_LENGTH = 2000;
    constexpr int MAX_BOARD_HEIGHT = 2 * MAX_WOOD_LENGTH;

    // Use a frequency array (histogram) to count occurrences of each wood length.
    // std::vector is efficient for sequential input reading and subsequent parallel read-only access.
    std::vector<int> wood_counts(MAX_WOOD_LENGTH + 1, 0);
    for (int i = 0; i < N; ++i) {
        int l;
        std::cin >> l;
        wood_counts[l]++;
    }

    // Capture the frequency array by const reference for safe, read-only access in the parallel lambda.
    const auto& const_wood_counts = wood_counts;

    // Use parlay::tabulate to compute the fence length for each possible board height in parallel.
    // The calculation for each height H is independent, making it a perfect use case for tabulate.
    auto board_counts = parlay::tabulate(MAX_BOARD_HEIGHT + 1, [&](size_t H_size) {
        int H = static_cast<int>(H_size);
        if (H < 2) { // Minimum board height is 1+1=2.
            return 0;
        }

        int current_fence_length = 0;
        // Optimization: if l1+l2=H and l2<=MAX_WOOD_LENGTH, then l1>=H-MAX_WOOD_LENGTH.
        // This prunes the search space for l1, significantly improving performance.
        int start_l1 = std::max(1, H - MAX_WOOD_LENGTH);
        
        // A split-loop micro-optimization is used to avoid a conditional inside the main loop.
        // Part 1: Sum contributions from pairs of different lengths (l1 < l2).
        // To avoid double counting, we iterate l1 up to (H-1)/2.
        for (int l1 = start_l1; l1 <= (H - 1) / 2; ++l1) {
            int l2 = H - l1;
            current_fence_length += std::min(const_wood_counts[l1], const_wood_counts[l2]);
        }
        
        // Part 2: Add contribution from pairs of the same length (l1 = l2 = H/2).
        // This is only possible if H is even.
        if (H % 2 == 0) {
            int l_equal = H / 2;
            // The check `l_equal >= start_l1` is implicitly true for valid H, so it is omitted.
            current_fence_length += const_wood_counts[l_equal] / 2;
        }
        
        return current_fence_length;
    });

    // We only care about heights from 2 upwards. A slice is an efficient, non-owning view.
    auto relevant_board_counts = parlay::make_slice(board_counts.begin() + 2, board_counts.end());

    // Use a single-pass map-reduce pattern to find the max and its count simultaneously.
    
    // Step 1: Create a delayed mapping from each fence length to a MaxFenceInfo struct.
    // parlay::delayed_map creates a lazy view, avoiding an intermediate array for efficiency.
    auto mapped_view = parlay::delayed_map(relevant_board_counts, [](int fence_length) {
        return MaxFenceInfo{fence_length, 1L};
    });

    // Step 2: Reduce the mapped view to a single result.
    // Define the identity element for our reduction monoid.
    // The initial maxLength must be less than any possible result (fence lengths are non-negative).
    MaxFenceInfo identity = {-1, 0L};

    // Create a monoid using our combiner function and identity element.
    auto max_info_monoid = parlay::make_monoid(combine_max_info, identity);

    // Perform the reduction to get the final result.
    MaxFenceInfo result = parlay::reduce(mapped_view, max_info_monoid);

    // The problem guarantees N>=2, so at least one board can be formed, and the result will be valid.
    std::cout << result.maxLength << " " << result.count << std::endl;

    return 0;
}