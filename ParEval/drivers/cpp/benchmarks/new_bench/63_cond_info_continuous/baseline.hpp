#ifdef USE_PARLAY
#include <parlay/primitives.h>
#endif

#pragma once
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>

using value = unsigned char;
const double encode_node_factor = 0.0;

// Helper from benchmark 3
double simple_entropy(std::vector<long> const& counts, long total) {
  if (total <= 0) return 0.0;
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
  Finds the best binary split for a continuous feature.
  'a_vals' are labels, 'b_vals' are feature values.
  'a_num' is num distinct labels, 'b_num' is num distinct feature values.
  Returns a pair: (minimum conditional entropy, best split value).
*/
std::pair<double, int> NO_INLINE correctCondInfoContinuous(std::vector<value> const& a_vals, int a_num,
                                                            std::vector<value> const& b_vals, int b_num) {
    size_t n = a_vals.size();
    if (n == 0) return {0.0, 0};
    
    std::vector<long> sums(a_num * b_num, 0);
    for (size_t i = 0; i < n; i++) {
        sums[a_vals[i] + b_vals[i] * a_num]++;
    }

    std::vector<long> low_counts(a_num, 0);
    std::vector<long> high_counts(a_num, 0);
    for (int i=0; i < b_num; i++) 
        for (int j=0; j < a_num; j++) high_counts[j] += sums[a_num*i + j];
    
    double cur_e = std::numeric_limits<double>::infinity();
    int cur_i = 0;
    long m = 0;
    for (int i = 0; i < b_num - 1; i++) {
        long split_total = 0;
        for (int j = 0; j < a_num; j++) {
            long count = sums[a_num*i + j];
            low_counts[j] += count;
            high_counts[j] -= count;
            split_total += count;
        }
        m += split_total;
        if (m > 0 && n - m > 0) {
            double e = simple_entropy(low_counts, m) + simple_entropy(high_counts, n - m);
            if (e < cur_e) {
                cur_e = e;
                cur_i = i + 1;
            }
        }
    }
    return std::pair(cur_e, cur_i);
}

#ifdef USE_PARLAY
// A correct parlay implementation for baseline.hpp
std::pair<double, int> NO_INLINE correctCondInfoContinuous(parlay::sequence<value> const& a_vals, int a_num,
                                                            parlay::sequence<value> const& b_vals, int b_num) {
    size_t n = a_vals.size();
    if (n == 0) return {0.0, 0};
    
    auto sums = parlay::histogram_by_index(parlay::delayed_tabulate(n, [&] (size_t i) {
			   return a_vals[i] + b_vals[i]*a_num;}), a_num * b_num);

    parlay::sequence<parlay::sequence<long>> hists(b_num);
    parlay::parallel_for(0, b_num, [&](size_t i) {
        hists[i] = sums.cut(i*a_num, (i+1)*a_num);
    });

    auto entropies = parlay::map(hists, [&](auto const& h) {
        long total = parlay::reduce(h);
        if (total == 0) return 0.0;
        double ecost = encode_node_factor * log2(float(1 + total));
        return ecost + parlay::reduce(parlay::delayed_map(h, [=] (long l) {
            return (l > 0) ? -(double)l * log2((double)l/total) : 0.0;
        }));
    });

    auto totals = parlay::map(hists, [&](auto const& h) { return parlay::reduce(h); });
    auto left_entropies = parlay::scan_inclusive(entropies, parlay::addm<double>());
    auto left_totals = parlay::scan_inclusive(totals, parlay::addm<long>());
    auto right_entropies = parlay::scan_exclusive(parlay::reverse(entropies), parlay::addm<double>());
    parlay::reverse_inplace(right_entropies);

    auto all_costs = parlay::tabulate(b_num - 1, [&](size_t i) {
        if (left_totals[i] > 0 && n - left_totals[i] > 0)
            return left_entropies[i] + right_entropies[i];
        return std::numeric_limits<double>::infinity();
    });
    
    size_t min_idx = parlay::min_element(all_costs) - all_costs.begin();
    return std::pair(all_costs[min_idx], min_idx + 1);
}
#endif