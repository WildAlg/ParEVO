#include <algorithm>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"

constexpr int max_val = 255;

struct Context {
    std::vector<value> s;
    size_t m;
};

void fillRand(std::vector<value> &s, size_t m) {
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distrib(0, m - 1);
    for (size_t i = 0; i < s.size(); i++) {
        s[i] = distrib(gen);
    }
}

void reset(Context *ctx) {
    fillRand(ctx->s, ctx->m);
    BCAST(ctx->s, CHAR);
}

Context *init() {
    Context *ctx = new Context();
    ctx->m = max_val;
    ctx->s.resize(DRIVER_PROBLEM_SIZE);
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    int maj = majority(ctx->s, ctx->m);
    (void)maj;
}

void NO_OPTIMIZE best(Context *ctx) {
    int maj = correctMajority(ctx->s, ctx->m);
    (void)maj;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 2048;
    const int TEST_MAX_VAL = 64;

    std::vector<value> s(TEST_SIZE);
    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int i = 0; i < numTries; i++) {
        fillRand(s, TEST_MAX_VAL);
        BCAST(s, CHAR);

        int correct = correctMajority(s, TEST_MAX_VAL);
        int test = majority(s, TEST_MAX_VAL);
        SYNC();
        
        bool isCorrect = true;
        if (IS_ROOT(rank) && correct != test) {
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