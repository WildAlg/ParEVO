#include <parlay/primitives.h>
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"

struct feature {
  bool discrete;
  int num;
  parlay::sequence<value> vals;
};

struct Context {
    feature a; // labels
    feature b; // features
};

void fillRand(parlay::sequence<value> &s, int num_vals) {
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distrib(0, num_vals - 1);
    for (size_t i = 0; i < s.size(); i++) {
        s[i] = distrib(gen);
    }
}

void reset(Context *ctx) {
    fillRand(ctx->a.vals, ctx->a.num);
    fillRand(ctx->b.vals, ctx->b.num);
    BCAST(ctx->a.vals, CHAR);
    BCAST(ctx->b.vals, CHAR);
}

Context *init() {
    Context *ctx = new Context();
    ctx->a = {true, 32};
    ctx->b = {false, 128}; // Continuous feature with 128 discrete bins
    ctx->a.vals.resize(DRIVER_PROBLEM_SIZE);
    ctx->b.vals.resize(DRIVER_PROBLEM_SIZE);
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    auto p = cond_info_continuous(ctx->a, ctx->b);
    (void)p;
}

void NO_OPTIMIZE best(Context *ctx) {
    auto p = correctCondInfoContinuous(ctx->a.vals, ctx->a.num, ctx->b.vals, ctx->b.num);
    (void)p;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 2048;
    
    feature a = {true, 16};
    feature b = {false, 32};
    a.vals.resize(TEST_SIZE);
    b.vals.resize(TEST_SIZE);

    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int i = 0; i < numTries; i++) {
        fillRand(a.vals, a.num);
        fillRand(b.vals, b.num);
        BCAST(a.vals, CHAR);
        BCAST(b.vals, CHAR);

        auto correct = correctCondInfoContinuous(a.vals, a.num, b.vals, b.num);
        auto test = cond_info_continuous(a, b);
        SYNC();
        
        bool isCorrect = true;
        if (IS_ROOT(rank) && (std::abs(correct.first - test.first) > 1e-6 || correct.second != test.second)) {
            isCorrect = false;
        }
        BCAST_PTR(&isCorrect, 1, CXX_BOOL);
        if (!isCorrect) return false;
    }
    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}