#ifdef USE_PARLAY
#include <parlay/primitives.h>
#endif

#pragma once
#include <vector>

/* Multiply the matrix A by the vector x. Store the results in the vector y.
   A is an MxN matrix stored in row-major, x has N elements, and y has M elements.
   Example:

   input: A=[[1, -1, 2], [0, -3, 1]] x=[2, 1, 0]
   output: y=[1, -3]
*/
void NO_INLINE correctGemv(std::vector<double> const& A, std::vector<double> const& x, std::vector<double> &y, size_t M, size_t N) {
   for (size_t i = 0; i < M; i++) {
      y[i] = 0;
      for (size_t j = 0; j < N; j++) {
         y[i] += A[i * N + j] * x[j];
      }
   }
}
#ifdef USE_PARLAY
void NO_INLINE correctGemv(parlay::sequence<double> const& A, parlay::sequence<double> const& x, parlay::sequence<double> &y, size_t M, size_t N) {
   for (size_t i = 0; i < M; i++) {
      y[i] = 0;
      for (size_t j = 0; j < N; j++) {
         y[i] += A[i * N + j] * x[j];
      }
   }
}
#endif
