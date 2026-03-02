#pragma once
#include <vector>
#include <numeric>
#include <parlay/primitives.h>

/* Return the logical XOR reduction of the vector of bools x.
   Example:

   input: [false, false, false, true]
   output: true
*/
bool correctReduceLogicalXOR(std::vector<bool> const& x) {
    return std::reduce(x.begin(), x.end(), false, [] (const auto &a, const auto &b) {
        return a != b;
    });
}

#ifdef USE_PARLAY
bool correctReduceLogicalXOR(parlay::sequence<bool> const& x) {
    return std::reduce(x.begin(), x.end(), false, [] (const auto &a, const auto &b) {
        return a != b;
    });
}
#endif
