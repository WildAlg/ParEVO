#include <iostream>
#include <array>
#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/delayed_sequence.h>
#include <parlay/monoid.h>

using namespace std;

// Names are < 1,000,000 < 2^20, so 20 bits are sufficient.
constexpr int MAX_BITS = 20;

// A type to hold the counts of set bits for all bit positions.
// The count for any bit position cannot exceed N (<= 1,000,000), so `int` is sufficient
// and more memory-efficient than `long long`.
using BitCounts = array<int, MAX_BITS>;

int main() {
    // Fast I/O for performance.
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    cin >> n;

    // If there are fewer than two residents, no friendships can be formed.
    if (n <= 1) {
        cout << 0 << endl;
        return 0;
    }

    // Read all resident names into a parlay sequence for parallel processing.
    parlay::sequence<int> names(n);
    for (int i = 0; i < n; ++i) {
        cin >> names[i];
    }

    // The total planet value is the sum of (a XOR b) for all pairs {a, b}.
    // By linearity, this can be calculated per bit:
    // Total Value = sum_{k=0..19} [ 2^k * (count of 0s at bit k) * (count of 1s at bit k) ]
    // We compute counts for all bit positions in a single parallel pass for data locality.

    // Define a monoid for element-wise addition of BitCounts arrays.
    BitCounts identity_counts{}; // Zero-initialized.
    auto bit_counts_add_monoid = parlay::make_monoid(
        [](const BitCounts& a, const BitCounts& b) {
            BitCounts result;
            for (int i = 0; i < MAX_BITS; ++i) {
                result[i] = a[i] + b[i];
            }
            return result;
        },
        identity_counts);

    // Create a delayed sequence that maps each name to its bit representation.
    // `delayed_map` is lazy, avoiding allocation of a large intermediate sequence.
    auto delayed_bits = parlay::delayed_map(names, [](int name) -> BitCounts {
        BitCounts counts{};
        for (int k = 0; k < MAX_BITS; ++k) {
            counts[k] = (name >> k) & 1;
        }
        return counts;
    });

    // Reduce in parallel to get the total counts of set bits for each position.
    BitCounts total_set_bits = parlay::reduce(delayed_bits, bit_counts_add_monoid);

    // Calculate the contribution of each bit to the total sum in parallel.
    auto contributions = parlay::tabulate(MAX_BITS, [&](size_t k) -> long long {
        // Cast to long long to prevent overflow in multiplication.
        long long count_1 = total_set_bits[k];
        long long count_0 = n - count_1;
        long long power_of_2 = 1LL << k;
        return count_0 * count_1 * power_of_2;
    });

    // Sum up the contributions from all bits in parallel.
    long long total_value = parlay::reduce(contributions);

    cout << total_value << endl;

    return 0;
}