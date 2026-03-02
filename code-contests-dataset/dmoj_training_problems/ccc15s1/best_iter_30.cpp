/*
 * Canadian Computing Competition: 2015 Stage 1, Senior #1
 * Solution using parlaylib for parallel computation.
 * 
 * Approach:
 * - Model the stack operations as function compositions.
 * - Process from right to left (reverse order).
 * - A '0' (pop) increases demand for elements to be popped.
 * - A number (push) satisfies demand or survives.
 * - Use parlay::scan to compute cumulative demand functions.
 * - Use parlay::reduce to sum survivors.
 * 
 * Optimizations:
 * - Use 'int' for State to minimize memory bandwidth (output fits in int, K <= 100,000).
 * - Branchless combine function for the monoid.
 * - Delayed sequences to avoid intermediate memory allocations.
 * - Pass-by-value for small State struct.
 */

#include <iostream>
#include <vector>
#include <algorithm>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// State representing the function f(x) = max(L, x + S).
// Tracks the propagation of "pop demand" from right to left.
struct State {
    int S;
    int L;
    
    // Sentinel for negative infinity.
    // Must be small enough so max(MIN_VAL, x) = x for valid x >= 0.
    // Must be large enough so MIN_VAL + S doesn't underflow.
    static constexpr int MIN_VAL = -1000000; 

    State() : S(0), L(MIN_VAL) {}
    State(int s, int l) : S(s), L(l) {}
};

// Monoid composition function: combine(a, b) -> f_B(f_A(x))
// 'a' is the state from the right (applied first).
// 'b' is the state from the left (applied second).
inline State combine(State a, State b) {
    return State(a.S + b.S, std::max(b.L, a.L + b.S));
}

int main() {
    // Optimize standard I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int K;
    if (!(cin >> K)) return 0;

    // Read input into a standard vector
    vector<int> input(K);
    for (int i = 0; i < K; ++i) {
        cin >> input[i];
    }

    // Create a delayed sequence of States corresponding to the reversed input.
    // Index i corresponds to input[K - 1 - i].
    auto states_view = parlay::delayed_tabulate(K, [&](size_t i) {
        int val = input[K - 1 - i];
        if (val == 0) {
            // '0' (Pop): Increases demand by 1.
            return State(1, State::MIN_VAL);
        } else {
            // Number (Push): Satisfies 1 demand unit.
            return State(-1, 0);
        }
    });

    // Compute exclusive prefix sums of the states.
    auto [prefixes, total] = parlay::scan(states_view, parlay::binary_op(combine, State()));

    // Calculate the sum of surviving numbers.
    long long survivor_sum = parlay::reduce(parlay::delayed_tabulate(K, [&](size_t i) -> long long {
        int val = input[K - 1 - i];
        
        if (val == 0) return 0;

        // Get the accumulated state from the right
        State st = prefixes[i];
        
        // Calculate demand: f(0) = max(L, 0 + S)
        int demand = std::max(st.L, st.S);

        if (demand == 0) {
            return val;
        } else {
            return 0;
        }
    }));

    cout << survivor_sum << endl;

    return 0;
}