#ifdef USE_PARLAY
#include <parlay/primitives.h>
#endif

#pragma once
#include <vector>
#include <numeric>
#include <algorithm>

using value = unsigned char;

/*
  Finds the most frequent value in a sequence of unsigned chars.
  's' is the sequence of values.
  'm' is the number of possible distinct values (i.e., the size of the histogram).
  Returns the value that appears most often.
*/
int NO_INLINE correctMajority(std::vector<value> const& s, size_t m) {
   if (s.empty()) return -1;
   std::vector<long> counts(m, 0);
   for (value x : s) {
      counts[x]++;
   }
   return std::max_element(counts.begin(), counts.end()) - counts.begin();
}

#ifdef USE_PARLAY
int NO_INLINE correctMajority(parlay::sequence<value> const& s, size_t m) {
   if (s.empty()) return -1;
   std::vector<long> counts(m, 0);
   for (value x : s) {
      counts[x]++;
   }
   return std::max_element(counts.begin(), counts.end()) - counts.begin();
}
#endif