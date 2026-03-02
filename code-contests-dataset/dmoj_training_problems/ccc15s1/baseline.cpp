#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

using namespace std;

// Represents the state of pending pop operations (zeros)
// propagating from right to left.
// The transformation is of the form f(x) = max(L, x + S)
// where x is the number of pending pops required by the suffix.
struct State {
    long long S;
    long long L;
    
    // A value representing negative infinity, safe for arithmetic operations
    static constexpr long long MIN_VAL = -1e16; 

    State() : S(0), L(MIN_VAL) {}
    State(long long s, long long l) : S(s), L(l) {}
};

// Monoid composition function
// Combines two states a and b (a applied first, then b)
// Corresponds to function composition f_B(f_A(x))
State combine(const State& a, const State& b) {
    long long s_new = a.S + b.S;
    // Calculate new L: max(L_B, f_B(L_A_effective))
    // Effectively: max(L_B, L_A + S_B) taking care of MIN_VAL
    long long term2 = (a.L == State::MIN_VAL) ? State::MIN_VAL : (a.L + b.S);
    long long l_new = std::max(b.L, term2);
    return State(s_new, l_new);
}

int main() {
    // Optimize standard I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int K;
    if (!(cin >> K)) return 0;

    // Read input sequence
    parlay::sequence<int> input(K);
    for (int i = 0; i < K; ++i) {
        cin >> input[i];
    }

    // Create a reversed view of the input to process right-to-left
    // This allows us to determine cancellation based on future '0's
    auto reversed_input = parlay::tabulate(K, [&](size_t i) {
        return input[K - 1 - i];
    });

    // Map each element to a State transformation
    // If val == 0 (pop): increases demand by 1. x -> x + 1. S=1, L=MIN
    // If val > 0 (push): consumes demand. x -> max(0, x - 1). S=-1, L=0
    auto states = parlay::map(reversed_input, [&](int val) {
        if (val == 0) {
            return State(1, State::MIN_VAL);
        } else {
            return State(-1, 0);
        }
    });

    // Perform exclusive scan to compute the cumulative state before each element
    // parlay::binary_op creates a Monoid from the combine function and identity State()
    auto [prefixes, total] = parlay::scan(states, parlay::binary_op(combine, State()));

    // Determine which numbers survive and sum them up
    // A number survives if the pending pop count 'cnt' from its right is 0
    auto survivors = parlay::tabulate(K, [&](size_t i) -> long long {
        int val = reversed_input[i];
        if (val == 0) return 0;

        // Apply the prefix state to the initial value 0
        // cnt = f(0) = max(L, 0 + S)
        State st = prefixes[i];
        long long current_cnt = std::max(st.L, st.S); 
        
        if (current_cnt == 0) {
            return val;
        } else {
            return 0;
        }
    });

    // Reduce to get the final sum
    long long result = parlay::reduce(survivors);

    cout << result << endl;

    return 0;
}