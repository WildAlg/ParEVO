#ifdef USE_PARLAY
#include <parlay/primitives.h>
#endif

#pragma once
#include <vector>

/* Compute the ReLU function on every element of x. Elements less than zero become zero,
   while elements greater than zero stay the same.
   Example:

   input: [-1.8, 24.0, 1.2, 0.0, -5.1, -0.2, 4.5]
   output: [0, 24.0, 1.2, 0, 0, 0, 4.5]
*/
void NO_INLINE correctRelu(std::vector<double> &x) {
    std::transform(x.begin(), x.end(), x.begin(), [](double v) {
        return std::max(0.0, v);
    });
}
#ifdef USE_PARLAY
void NO_INLINE correctRelu(parlay::sequence<double> &x) {
    std::transform(x.begin(), x.end(), x.begin(), [](double v) {
        return std::max(0.0, v);
    });
}
#endif
