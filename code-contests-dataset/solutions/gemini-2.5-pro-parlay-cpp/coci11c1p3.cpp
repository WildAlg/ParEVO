#include <iostream>
#include <algorithm> // For std::max
#include <limits>    // For std::numeric_limits
#include <vector>    // For completeness, though not strictly required by the logic

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/delayed_sequence.h>

// A helper function to determine the number of bits required to represent a number n.
// This is equivalent to floor(log2(n)) + 1 for n > 0.
int get_active_bits(unsigned int n) {
    if (n == 0) {
        return 0;
    }
    // Use fast compiler intrinsics where available (GCC, Clang).
    // __builtin_clz counts the number of leading zeros in an unsigned integer.
    #if defined(__GNUC__) || defined(__clang__)
    return std::numeric_limits<unsigned int>::digits - __builtin_clz(n);
    #else
    // A portable fallback implementation for other compilers.
    int num_bits = 0;
    while (n > 0) {
        n >>= 1;
        num_bits++;
    }
    return num_bits;
    #endif
}

int main() {
    // Fast I/O for performance in competitive programming.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int n;
    std::cin >> n;

    // If there are fewer than 2 residents, no pairs can be formed,
    // so the total planet value is 0.
    if (n <= 1) {
        // We must still consume the input names to avoid issues with
        // the judging environment reading subsequent test cases.
        int temp_name;
        for (int i = 0; i < n; ++i) {
            std::cin >> temp_name;
        }
        std::cout << 0 << std::endl;
        return 0;
    }

    // Read names into a parlay::sequence for parallel processing.
    // Simultaneously, find the maximum name value to optimize the number of bits to check.
    parlay::sequence<int> names(n);
    int max_name = 0;
    for (int i = 0; i < n; ++i) {
        std::cin >> names[i];
        max_name = std::max(max_name, names[i]);
    }

    // Determine the number of bits required to represent the max name.
    // This avoids iterating over unnecessary higher-order bits where all names are 0.
    const int num_bits = get_active_bits(static_cast<unsigned int>(max_name));

    // The core of the algorithm is to calculate the contribution of each bit position
    // to the total sum and then add them up. This exploits the linearity of summation.
    // TotalSum = sum_over_bits_k (contribution of bit k)
    // Contribution(k) = (number of pairs with different k-th bit) * 2^k
    //                 = (count_0s_at_k * count_1s_at_k) * 2^k

    // parlay::delayed_tabulate creates a "lazy" or "delayed" sequence. The values
    // are not computed and stored in memory all at once. Instead, the generating
    // function is called as needed by the consuming operation (in this case, parlay::reduce).
    // This avoids allocating an intermediate array for the contributions, which can
    // improve performance by reducing memory traffic and fusing the computation.
    auto bit_contributions = parlay::delayed_tabulate(num_bits, [&](size_t k) -> long long {
        // For each bit 'k', count how many names have this bit set.
        // parlay::count_if is an efficient parallel primitive for this reduction.
        long long count1 = parlay::count_if(names, [&](int name) {
            return (name >> k) & 1;
        });

        // The number of names with bit 'k' not set is simply n - count1.
        long long count0 = n - count1;

        // Calculate the contribution for this bit.
        // Use 1LL to promote 1 to a long long before shifting, preventing overflow
        // for large k, as the result can exceed 2^31.
        return count0 * count1 * (1LL << k);
    });

    // Sum up the contributions from all bit positions using a parallel reduction.
    // The reduction is performed directly on the delayed sequence, fusing the
    // generation and reduction steps into a single, efficient pipeline.
    long long total_planet_value = parlay::reduce(bit_contributions);

    std::cout << total_planet_value << std::endl;

    return 0;
}