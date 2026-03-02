#ifdef USE_PARLAY
#include <parlay/primitives.h>
#endif

#pragma once

#include <numeric>
#include <vector>

/* Compute the prefix sum of the vector x into output.
   Example:

   input: [1, 7, 4, 6, 6, 2]
   output: [1, 8, 12, 18, 24, 26]
*/
void NO_INLINE correctPrefixSum(std::vector<double> const& x, std::vector<double> &output) {
    std::inclusive_scan(x.begin(), x.end(), output.begin());
}

#ifdef USE_PARLAY
void NO_INLINE correctPrefixSum(parlay::sequence<double> const& x, parlay::sequence<double> &output) {
    std::inclusive_scan(x.begin(), x.end(), output.begin());
}
#endif
