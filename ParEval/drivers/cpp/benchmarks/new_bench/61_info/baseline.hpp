#ifdef USE_PARLAY
#include <parlay/primitives.h>
#endif

#pragma once
#include <vector>
#include <numeric>
#include <cmath>

using value = unsigned char;
const double encode_node_factor = 0.0; // From original code

/*
  Calculates the information content (entropy * total) of a sequence.
  's' is the sequence of values (e.g., class labels).
  'num_vals' is the number of possible distinct values.
  Returns the information content as a double.
*/
double NO_INLINE correctInfo(std::vector<value> const& s, int num_vals) {
   size_t n = s.size();
   if (n == 0) return 0.0;
   
   std::vector<long> counts(num_vals, 0);
   for (value x : s) {
      counts[x]++;
   }
   
   double total_entropy = 0;
   for (long count : counts) {
       if (count > 0) {
           total_entropy -= (double)count * log2((double)count / n);
       }
   }
   // Add optional factor to prevent overfitting
   total_entropy += encode_node_factor * log2(float(1 + n));
   return total_entropy;
}

#ifdef USE_PARLAY
double NO_INLINE correctInfo(parlay::sequence<value> const& s, int num_vals) {
   size_t n = s.size();
   if (n == 0) return 0.0;
   auto counts = parlay::histogram_by_index(s, num_vals);
   
   double total_entropy = parlay::reduce(parlay::delayed_map(counts, [=] (long count) {
      return (count > 0) ? -(double)count * log2((double)count / n) : 0.0;
   }));
   total_entropy += encode_node_factor * log2(float(1 + n));
   return total_entropy;
}
#endif