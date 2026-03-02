#include <iostream>
#include <vector>
#include <numeric>
#include <utility>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/monoid.h>

int main() {
    // Use fast I/O for performance
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int K;
    std::cin >> K;

    if (K == 0) {
        std::cout << 0 << std::endl;
        return 0;
    }

    // Read all K operations into a parlay sequence.
    // Use uninitialized memory for speed since we immediately overwrite it.
    auto ops = parlay::sequence<int>::uninitialized(K);
    for (int i = 0; i < K; ++i) {
        std::cin >> ops[i];
    }

    // This parallel strategy decomposes the problem by stack depth.
    // Operations at different depths are independent and can be processed in parallel.

    // Step 1: Compute stack depth changes. A positive number is a push (+1), a zero is a pop (-1).
    auto depth_deltas = parlay::map(ops, [](int op) {
        return (op > 0) ? 1 : -1;
    });

    // Step 2: Compute the prefix sum of depth changes to find the depth after each operation.
    parlay::scan_inclusive_inplace(depth_deltas, parlay::plus<int>());
    const auto& depths_after_op = depth_deltas;

    // Step 3: Create events, mapping each operation to its effective stack depth.
    // An event is a pair of (depth, original_index).
    auto events = parlay::tabulate(K, [&](size_t i) {
        int effective_depth;
        if (ops[i] > 0) {
            // A push operation occurs at the depth *after* the push.
            effective_depth = depths_after_op[i];
        } else { // ops[i] == 0, a pop operation
            // A pop cancels an item at the depth *before* the pop.
            // The problem guarantees the first operation is a push, so i > 0 for a pop.
            effective_depth = depths_after_op[i-1];
        }
        return std::make_pair(effective_depth, i);
    });

    // Step 4: Group events by depth.
    // First, sort events by depth. parlay::sort is stable, preserving original time order for the same depth.
    auto sorted_events = parlay::sort(events);
    
    // `group_by_key` on a sorted sequence of pairs groups values by key.
    // The result is a sequence of pairs: (key, sequence_of_values)
    auto groups_by_depth = parlay::group_by_key(sorted_events);

    // Step 5: Process each group in parallel to find the sum of uncancelled numbers.
    // This is a hybrid parallel-sequential approach.
    auto group_sums = parlay::map(groups_by_depth, [&](const auto& group) {
        // `group` is a pair of (depth, sequence_of_original_indices)
        const auto& indices_in_group = group.second;
        
        // Within each depth group, simulate the LIFO stack sequentially.
        // We maintain a running sum for efficiency, avoiding a second pass.
        std::vector<int> value_stack;
        long long current_sum = 0;

        for (size_t original_index : indices_in_group) {
            if (ops[original_index] > 0) {
                int val = ops[original_index];
                value_stack.push_back(val);
                current_sum += val;
            } else {
                // The problem guarantees we won't pop from an empty stack within a valid sequence.
                if (!value_stack.empty()) {
                    current_sum -= value_stack.back();
                    value_stack.pop_back();
                }
            }
        }
        return current_sum;
    });

    // Step 6: Sum up the results from all groups for the final answer.
    long long total_sum = parlay::reduce(group_sums);
    std::cout << total_sum << std::endl;

    return 0;
}