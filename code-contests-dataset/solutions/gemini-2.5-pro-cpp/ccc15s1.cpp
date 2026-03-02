#include <iostream>
#include <vector>
#include <numeric>
#include <utility>
#include <algorithm>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/alloc.h>

// A 'Result' represents the net effect of processing a sub-sequence of operations.
// It consists of two parts:
// 1. first: The count of 'pop' operations (from input '0') that could not be
//    satisfied within the sub-sequence because the local stack was empty. These
//    are "unmatched pops".
// 2. second: A sequence of numbers from 'push' operations (positive inputs) that were not
//    cancelled by 'pop' operations within the sub-sequence. These are the
//    "remaining pushes".
using Result = std::pair<size_t, parlay::sequence<int>>;

// The 'combine' function is an associative binary operator for the parallel reduction.
// It merges the results from two adjacent sub-sequences, 'left' and 'right', to
// produce a single result representing their combined effect. This implementation
// is branchless for potential performance benefits.
Result combine(const Result& left, const Result& right) {
    // Pops from the right subsequence cancel pushes from the left.
    // `matched` is the number of such cancellations.
    size_t matched = std::min(left.second.size(), right.first);
    
    // The number of pushes from the left that survive.
    size_t keep_left_count = left.second.size() - matched;
    
    // The new number of unmatched pops is the sum of pops from both sides,
    // minus the ones that were just matched.
    size_t new_pops = left.first + right.first - matched;
    
    // The new sequence of pushes consists of the surviving pushes from the left,
    // followed by all pushes from the right. A new sequence is allocated to
    // hold the combined result.
    auto new_pushes = parlay::sequence<int>::uninitialized(keep_left_count + right.second.size());
    
    // Efficiently copy the surviving part of left's pushes and all of right's pushes
    // into the newly allocated sequence. These copies can run in parallel.
    parlay::copy(left.second.head(keep_left_count), new_pushes.head(keep_left_count));
    parlay::copy(right.second, new_pushes.tail(right.second.size()));
    
    return {new_pops, std::move(new_pushes)};
}

int main() {
    // Fast I/O for competitive programming.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int K;
    std::cin >> K;

    // Handle the edge case of no inputs.
    if (K == 0) {
        std::cout << 0 << std::endl;
        return 0;
    }

    // Read all K inputs into a parlay sequence for efficient parallel processing.
    parlay::sequence<int> inputs(K);
    for (int i = 0; i < K; ++i) {
        std::cin >> inputs[i];
    }
    
    // Set a block size for parallel processing. This is a performance tuning parameter.
    // Processing in blocks amortizes the overhead of parallel task creation and
    // improves cache locality by performing sequential work on contiguous data.
    const size_t block_size = 4096;
    size_t num_blocks = (K + block_size - 1) / block_size;

    // Map phase: process each block of input in parallel to get a partial Result.
    // parlay::tabulate creates a sequence by running a function for each index
    // [0, num_blocks) in parallel. This is a standard way to implement blocked algorithms.
    auto block_results = parlay::tabulate(num_blocks, [&](size_t i) {
        size_t start = i * block_size;
        size_t end = std::min(static_cast<size_t>(K), (i + 1) * block_size);
        
        // Use a standard vector as a temporary stack for efficient sequential
        // processing within the block.
        std::vector<int> local_stack;
        size_t pops = 0; // Count of unmatched pops for this block.
        
        for (size_t j = start; j < end; ++j) {
            if (inputs[j] > 0) {
                // A positive number is a 'push'.
                local_stack.push_back(inputs[j]);
            } else { // inputs[j] == 0, representing a 'pop' operation.
                if (!local_stack.empty()) {
                    local_stack.pop_back();
                } else {
                    // This 'pop' cannot be satisfied within this block. Record it.
                    pops++;
                }
            }
        }
        // The result for this block is the number of unmatched pops
        // and the remaining pushes, converted to a parlay::sequence.
        return std::make_pair(pops, parlay::to_sequence(local_stack));
    });

    // Reduce phase: combine the results from all blocks into a single final result.
    // The identity for the reduction is 0 pops and an empty sequence of pushes,
    // as this has no effect when combined with another Result.
    Result identity = {0, parlay::sequence<int>()};
    Result final_result = parlay::reduce(block_results, parlay::make_monoid(combine, identity));
    
    // The problem statement guarantees that globally, there are at least as many
    // positive numbers as zeros. This implies the final number of unmatched pops
    // will be 0. The final list of numbers to be summed is in final_result.second.
    // Sum these numbers in parallel using another parlay::reduce.
    long long total_sum = parlay::reduce(final_result.second, parlay::addm<long long>());

    std::cout << total_sum << std::endl;

    return 0;
}