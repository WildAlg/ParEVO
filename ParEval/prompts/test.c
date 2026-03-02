#include <parlay/primitives.h>

/* Factorize the matrix A into A=LU where L is a lower triangular matrix and 
   U is an upper triangular matrix. Store the results for L and U into the 
   original matrix A. A is an NxN matrix stored in row-major.
   Use ParlayLib to compute in parallel.
   Example:
   input: [[4, 3], [6, 3]]
   output: [[4, 3], [1.5, -1.5]]
*/
void luFactorize(parlay::sequence<double> &A, size_t N) {



// /* Factorize the matrix A into A=LU where L is a lower triangular matrix and 
//    U is an upper triangular matrix. Store the results for L and U into the 
//    original matrix A. A is an NxN matrix stored in row-major.
//    Translate the OpenMP inplementation to ParlayLib to compute in parallel.
//    Example:
//    input: [[4, 3], [6, 3]]
//    output: [[4, 3], [1.5, -1.5]]
// */
// OpenMP implementation of luFactorize
// // #include <omp.h>
// //
// // void luFactorize(std::vector<double> &A, size_t N) {
// //     for (size_t k = 0; k < N; ++k) {
// //         #pragma omp parallel for
// //         for (size_t i = k + 1; i < N; ++i) {
// //             double factor = A[i * N + k] / A[k * N + k];
// //             A[i * N + k] = factor;
// //             for (size_t j = k + 1; j < N; ++j) {
// //                 A[i * N + j] -= factor * A[k * N + j];
// //             }
// //         }
// //     }
// // }
// // ParlayLib implementation of luFactorize
// #include <parlay/primitives.h>
void luFactorize(parlay::sequence<double> &A, size_t N) {
