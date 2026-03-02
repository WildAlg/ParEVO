#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <algorithm>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/io.h>

// Function to convert char to int (J=0, O=1, I=2)
int char_to_int(char c) {
    if (c == 'J') return 0;
    if (c == 'O') return 1;
    return 2;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int n;
    std::cin >> n;

    std::string sa_str, sb_str, sc_str;
    std::cin >> sa_str >> sb_str >> sc_str;

    auto s_a = parlay::map(sa_str, char_to_int);
    auto s_b = parlay::map(sb_str, char_to_int);
    auto s_c = parlay::map(sc_str, char_to_int);

    // Generate 9 possible reachable base sequences
    parlay::sequence<parlay::sequence<int>> base_sequences(9);
    parlay::parallel_for(0, 9, [&](int k) {
        int kb = k / 3;
        int kc = k % 3;
        int ka = (1 - kb - kc + 3) % 3;
        
        base_sequences[k] = parlay::sequence<int>(n);
        parlay::parallel_for(0, n, [&](int i) {
            base_sequences[k][i] = (ka * s_a[i] + kb * s_b[i] + kc * s_c[i]) % 3;
        });
    });

    // Precompute prefix sums for each base sequence and character
    auto prefix_sums = parlay::sequence<parlay::sequence<parlay::sequence<int>>>(
        9, parlay::sequence<parlay::sequence<int>>(3, parlay::sequence<int>(n + 1, 0)));
    
    // Flattened parallel loop for better load balancing
    parlay::parallel_for(0, 27, [&](size_t job_id){
        int k = job_id / 3;
        int c = job_id % 3;
        auto indicator = parlay::map(base_sequences[k], [&](int val) { return (int)(val == c); });
        parlay::scan_inclusive_inplace(indicator);
        prefix_sums[k][c][0] = 0;
        parlay::copy(indicator, prefix_sums[k][c].cut(1, n + 1));
    });

    auto count_char_in_range = [&](int k, int c, int l, int r) {
        if (l > r) return 0;
        return prefix_sums[k][c][r + 1] - prefix_sums[k][c][l];
    };

    int q;
    std::cin >> q;

    std::string t0_str;
    std::cin >> t0_str;
    
    // Represent T as intervals of same characters using a map
    std::map<int, int> t_intervals;
    if (n > 0) {
        auto t0 = parlay::map(t0_str, char_to_int);
        t_intervals[0] = t0[0];
        for (int i = 1; i < n; ++i) {
            if (t0[i] != t0[i - 1]) {
                t_intervals[i] = t0[i];
            }
        }
    }

    parlay::sequence<long long> mismatch_counts(9, (long long)n);
    if (n > 0) {
        for (auto const& [start_idx, val] : t_intervals) {
            auto it = t_intervals.upper_bound(start_idx);
            int end_idx = (it == t_intervals.end()) ? n - 1 : it->first - 1;
            for (int k = 0; k < 9; ++k) {
                long long matches = count_char_in_range(k, val, start_idx, end_idx);
                mismatch_counts[k] -= matches;
            }
        }
    }


    auto check_and_print = [&]() {
        bool possible = parlay::any_of(mismatch_counts, [](long long m) { return m == 0; });
        std::cout << (possible ? "Yes\n" : "No\n");
    };

    check_and_print();

    auto split_at = [&](int p) {
        if (p <= 0 || p >= n) return;
        auto it = t_intervals.upper_bound(p);
        if (it != t_intervals.begin()) {
            auto prev_it = std::prev(it);
            if (prev_it->first < p) {
                t_intervals[p] = prev_it->second;
            }
        }
    };

    // Process queries
    for (int j = 0; j < q; ++j) {
        int l_in, r_in;
        char c_char;
        std::cin >> l_in >> r_in >> c_char;
        
        int l = l_in - 1;
        int r = r_in - 1;
        int c_val = char_to_int(c_char);

        split_at(l);
        split_at(r + 1);

        auto it_start = t_intervals.find(l);
        auto it_end = t_intervals.find(r + 1);

        for (auto it = it_start; it != it_end; ++it) {
            int start = it->first;
            int old_val = it->second;
            auto next_it = std::next(it);
            int end = (next_it == it_end) ? r : next_it->first - 1;
            
            for (int k = 0; k < 9; ++k) {
                long long old_matches = count_char_in_range(k, old_val, start, end);
                long long new_matches = count_char_in_range(k, c_val, start, end);
                mismatch_counts[k] += old_matches - new_matches;
            }
        }
        
        if(it_start != it_end) t_intervals.erase(it_start, it_end);
        t_intervals[l] = c_val;

        auto current_it = t_intervals.find(l);
        if (current_it != t_intervals.begin()) {
            auto prev_it = std::prev(current_it);
            if (prev_it->second == current_it->second) {
                t_intervals.erase(current_it);
                current_it = prev_it;
            }
        }
        
        auto next_it = std::next(current_it);
        if (next_it != t_intervals.end() && current_it->second == next_it->second) {
            t_intervals.erase(next_it);
        }

        check_and_print();
    }

    return 0;
}