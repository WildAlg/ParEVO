#ifdef USE_PARLAY
#include <parlay/primitives.h>
#endif

#pragma once
#include <vector>

/* Compute z = alpha*x+y where x and y are vectors. Store the result in z.
   Example:
   
   input: x=[1, -5, 2, 9] y=[0, 4, 1, -1] alpha=2
   output: z=[2, -6, 5, 17]
*/
void NO_INLINE correctAxpy(double alpha, std::vector<double> const& x, std::vector<double> const& y, std::vector<double> &z) {
   for (size_t i = 0; i < x.size(); i += 1) {
      z[i] = alpha*x[i] + y[i];
   }
}
#ifdef USE_PARLAY
void NO_INLINE correctAxpy(double alpha, parlay::sequence<double> const& x, parlay::sequence<double> const& y, parlay::sequence<double> &z) {
   for (size_t i = 0; i < x.size(); i += 1) {
      z[i] = alpha*x[i] + y[i];
   }
}
#endif
