#include <cstddef> 
#include <cstdint>
#include <random>

#include "solution.hpp"

class my_lcg {
public:
    using result_type = uint32_t;

private:
    result_type state;

public:
    static constexpr result_type multiplier = 16807;
    static constexpr result_type modulus    = 2147483647; // 2^31 - 1

    // Требование: должен быть конструктор от seed
    explicit my_lcg(result_type seed = 1) : state(seed) {}

    // Требование: min() и max()
    static constexpr result_type min() { return 1; }
    static constexpr result_type max() { return modulus - 1; }

    // Требование: operator()
    result_type operator()() {
        state = (uint64_t(state) * multiplier) % modulus;
        return state;
    }
};


Status generate_bits(size_t n, uint32_t seed, uint32_t* result) {
    my_lcg gen(seed);
    for (size_t i = 0; i < n; ++i) {
        result[i] = gen();
    }
    return STATUS_OK;
}

Status generate_uniform(size_t n, uint32_t seed, float min, float max, float* result) {
    my_lcg gen(seed);
    std::uniform_real_distribution<float> d{min, max};

    for (size_t i = 0; i < n; ++i) {
        result[i] = d(gen);
    }
    return STATUS_OK;
}

Status generate_norm(size_t n, uint32_t seed, float mean, float stddev, float* result) {
    my_lcg gen(seed);
    std::normal_distribution<float> d{mean, stddev};

    for (size_t i = 0; i < n; ++i) {
        result[i] = d(gen);
    }

    return STATUS_OK;
}

Status generate_exponential(size_t n, uint32_t seed, float lambda, float* result) {
    my_lcg gen(seed);
    std::exponential_distribution<float> d{lambda};

    for (size_t i = 0; i < n; ++i) {
        result[i] = d(gen);
    }

    return STATUS_OK;
}

Status generate_bernoulli(size_t n, uint32_t seed, float probability, float* result) {
    my_lcg gen(seed);
    std::bernoulli_distribution d{probability};

    for (size_t i = 0; i < n; ++i) {
        result[i] = d(gen);
    }

    return STATUS_OK;
}
