#include <cstddef> 
#include <cstdint>
#include <random>

#include "solution.hpp"
#include "own_gen.cpp"

inline float uint32_to_float(uint32_t x, float min, float max) {
    return min + (max - min) * (x / 2147483647.0f); // modulus - 1
}

Status generate_bits(size_t n, uint32_t seed, uint32_t* result) {
    my_lcg gen(seed);
    for (size_t i = 0; i < n; ++i) {
        result[i] = gen();
    }
    return STATUS_OK;
}

Status generate_uniform(size_t n, uint32_t seed, float min, float max, float* result) {
    my_lcg gen(seed);

    for (size_t i = 0; i < n; ++i) {
        result[i] = uint32_to_float(gen(), min, max);
    }
    return STATUS_OK;
}

Status generate_norm(size_t n, uint32_t seed, float mean, float stddev, float* result) {
    my_lcg gen(seed);

    for (size_t i = 0; i < n; ++i) {
        float u1 = uint32_to_float(gen(), 0.f, 1.f);
        float u2 = uint32_to_float(gen(), 0.f, 1.f);
        float r = sqrtf(-2.f * logf(u1));
        float theta = 2.f * 3.14159265359f * u2;
        result[i] = mean + stddev * (r * cosf(theta));
    }

    return STATUS_OK;
}

Status generate_exponential(size_t n, uint32_t seed, float lambda, float* result) {
    my_lcg gen(seed);

    for (size_t i = 0; i < n; ++i) {
        float u = uint32_to_float(gen(), 0.f, 1.f);
        result[i] = -logf(1.f - u) / lambda;
    }

    return STATUS_OK;
}

Status generate_bernoulli(size_t n, uint32_t seed, float probability, float* result) {
    my_lcg gen(seed);

    for (size_t i = 0; i < n; ++i) {
        result[i] = (gen() < probability * my_lcg::max()) ? 1.f : 0.f;
    }

    return STATUS_OK;
}
