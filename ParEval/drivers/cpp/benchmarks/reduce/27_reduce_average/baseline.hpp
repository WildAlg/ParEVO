#ifdef USE_PARLAY
#include <parlay/primitives.h>
#endif

#pragma once
#include <vector>
#include <numeric>

/* Return the average of the vector x.
   Examples:

   input: [1, 8, 4, 5, 1]
   output: 3.8

   input: [2, 2, 2, 3]
   output: 2.25
*/
double NO_INLINE correctAverage(std::vector<double> const& x) {
    return std::reduce(x.begin(), x.end(), 0.0) / (double) x.size();
}

#ifdef USE_PARLAY
double NO_INLINE correctAverage(parlay::sequence<double> const& x) {
    return std::reduce(x.begin(), x.end(), 0.0) / (double) x.size();
}
#endif
