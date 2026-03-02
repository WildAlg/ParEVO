#ifdef USE_PARLAY
#include <parlay/primitives.h>
#endif

#pragma once
#include <vector>
#include <numeric>
#include <cmath>

using value = unsigned char;
const double encode_node_factor = 0.0;

// Helper to calculate entropy for a single histogram
double simple_entropy(std::vector<long> const& counts, long total) {
  double ecost = encode_node_factor * log2(float(1 + total));
  double total_entropy = 0.0;
  for (long l : counts) {
      if (l > 0) {
          total_entropy -= (double)l * log2((double)l/total);
      }
  }
  return ecost + total_entropy;
}


/*
  Calculates the information of 'a' conditioned on 'b' for discrete features.
  'a_vals' are the label values, 'b_vals' are the feature values.
  'a_num' is the number of distinct labels, 'b_num' is the number of distinct feature values.
*/
double NO_INLINE correctCondInfoDiscrete(std::vector<value> const& a_vals, int a_num,
                                          std::vector<value> const& b_vals, int b_num) {
    size_t n = a_vals.size();
    if (n == 0) return 0.0;
    
    std::vector<long> sums(a_num * b_num, 0);
    for (size_t i = 0; i < n; i++) {
        sums[a_vals[i] + b_vals[i] * a_num]++;
    }

    double total_cond_entropy = 0.0;
    for (int i = 0; i < b_num; i++) {
        std::vector<long> sub_hist;
        long sub_total = 0;
        for (int j = 0; j < a_num; j++) {
            long count = sums[j + i * a_num];
            sub_hist.push_back(count);
            sub_total += count;
        }
        if (sub_total > 0) {
            total_cond_entropy += simple_entropy(sub_hist, sub_total);
        }
    }
    return total_cond_entropy;
}

#ifdef USE_PARLAY
// Parlay version can be more complex to write from scratch without primitives.
// A direct translation is provided for API consistency.
double NO_INLINE correctCondInfoDiscrete(parlay::sequence<value> const& a_vals, int a_num,
                                          parlay::sequence<value> const& b_vals, int b_num) {
    size_t n = a_vals.size();
    if (n == 0) return 0.0;
    
    auto sums = parlay::histogram_by_index(parlay::delayed_tabulate(n, [&] (size_t i) {
			   return a_vals[i] + b_vals[i]*a_num;}), a_num * b_num);
               
    return parlay::reduce(parlay::tabulate(b_num, [&] (size_t i) {
        auto x = sums.cut(i*a_num, (i+1)*a_num);
        long total = parlay::reduce(x);
        if (total == 0) return 0.0;
        double ecost = encode_node_factor * log2(float(1 + total));
        return ecost + parlay::reduce(parlay::delayed_map(x, [=] (long l) {
            return (l > 0) ? -(double)l * log2((double)l/total) : 0.0;
        }));
    }));
}
#endif