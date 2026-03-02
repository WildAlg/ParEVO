#include <parlay/primitives.h>
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"

constexpr int max_val = 255;

struct Context {
    parlay::sequence<value> s;
    int num_vals;
};

void fillRand(parlay::sequence<value> &s, int num_vals) {
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distrib(0, num_vals - 1);
    for (size_t i = 0; i < s.size(); i++) {
        s[i] = distrib(gen);
    }
}

void reset(Context *ctx) {
    fillRand(ctx->s, ctx->num_vals);
    BCAST(ctx->s, CHAR);
}

Context *init() {
    Context *ctx = new Context();
    ctx->num_vals = max_val;
    ctx->s.resize(DRIVER_PROBLEM_SIZE);
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    double i = info(ctx->s, ctx->num_vals);
    (void)i;
}

void NO_OPTIMIZE best(Context *ctx) {
    double i = correctInfo(ctx->s, ctx->num_vals);
    (void)i;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 2048;
    const int TEST_MAX_VAL = 64;

    parlay::sequence<value> s(TEST_SIZE);
    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int i = 0; i < numTries; i++) {
        fillRand(s, TEST_MAX_VAL);
        BCAST(s, CHAR);

        double correct = correctInfo(s, TEST_MAX_VAL);
        double test = info(s, TEST_MAX_VAL);
        SYNC();
        
        bool isCorrect = true;
        if (IS_ROOT(rank) && std::abs(correct - test) > 1e-6) {
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