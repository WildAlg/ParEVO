#ifdef USE_PARLAY
#include <parlay/primitives.h>
#endif

#pragma once
#include <vector>

/* Replace every element of x with the square of its value.
   Example:

   input: [5, 1, 2, -4, 8]
   output: [25, 1, 4, 16, 64]
*/
void NO_INLINE correctSquareEach(std::vector<int> &x) {
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = x[i] * x[i];
    }
}
#ifdef USE_PARLAY
void NO_INLINE correctSquareEach(parlay::sequence<int> &x) {
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = x[i] * x[i];
    }
}
#endif
