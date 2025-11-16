#include <cstddef> 
#include <cstdint>
#include <random>
#include <omp.h>

#include "solution.hpp"
#include "own_gen.cpp"

uint32_t skip_ahead(uint32_t seed, uint64_t k) {
    // вычислить a^k mod m
    uint64_t a = my_lcg::multiplier;
    uint64_t m = my_lcg::modulus;

    uint64_t ak = 1;
    while (k) {
        if (k & 1) ak = (ak * a) % m;
        a = (a * a) % m;
        k >>= 1;
    }
    return (seed * ak) % m;
}

inline float uint32_to_float(uint32_t x, float min, float max) {
    return min + (max - min) * (x / 2147483647.0f); // modulus - 1
}

Status generate_bits(size_t n, uint32_t seed, uint32_t* result) {
    size_t T = omp_get_max_threads();
    size_t block = n / T;

    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        size_t start = t * block;
        size_t end   = (t == T-1 ? n : start + block);

        // вычисляем стартовое состояние для потока
        uint32_t thread_seed = skip_ahead(seed, start);

        my_lcg gen(thread_seed);

        for (size_t i = start; i < end; ++i) {
            result[i] = gen();
        }
    }
    return STATUS_OK;
}

Status generate_uniform(size_t n, uint32_t seed, float min, float max, float* result) {
    size_t T = omp_get_max_threads();
    size_t block = n / T;

    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        size_t start = t * block;
        size_t end = (t == T - 1 ? n : start + block);

        uint32_t thread_seed = skip_ahead(seed, start);
        my_lcg gen(thread_seed);

        float scale = (max - min) / float(my_lcg::max());

        for (size_t i = start; i < end; ++i) {
            uint32_t u = gen();
            result[i] = min + scale * float(u);
        }
    }
    return STATUS_OK;
}

Status generate_norm(size_t n, uint32_t seed, float mean, float stddev, float* result) {
    size_t T = omp_get_max_threads();
    size_t block = n / T;

    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        size_t start = t * block;
        size_t end   = (t == T - 1 ? n : start + block);

        uint32_t thread_seed = skip_ahead(seed, start);
        my_lcg gen(thread_seed);

        for (size_t i = start; i < end; i += 2) {
            // uniform(0,1)
            float u1 = uint32_to_float(gen(), 0.0f, 1.0f);
            float u2 = uint32_to_float(gen(), 0.0f, 1.0f);

            // Box-Muller
            float r = sqrtf(-2.0f * logf(u1));
            float theta = 6.28318530718f * u2; // 2π

            float z0 = r * cosf(theta);
            float z1 = r * sinf(theta);

            result[i] = mean + stddev * z0;
            if (i + 1 < end)
                result[i + 1] = mean + stddev * z1;
        }
    }

    return STATUS_OK;
}

Status generate_exponential(size_t n, uint32_t seed, float lambda, float* result) {
    size_t T = omp_get_max_threads();
    size_t block = n / T;

    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        size_t start = t * block;
        size_t end   = (t == T - 1 ? n : start + block);

        uint32_t thread_seed = skip_ahead(seed, start);
        my_lcg gen(thread_seed);

        float inv_lambda = 1.0f / lambda;

        for (size_t i = start; i < end; ++i) {
            float u = uint32_to_float(gen(), 0.0f, 1.0f);

            result[i] = -logf(1.0f - u) * inv_lambda;
        }
    }

    return STATUS_OK;
}

Status generate_bernoulli(size_t n, uint32_t seed, float probability, float* result) {
    size_t T = omp_get_max_threads();
    size_t block = n / T;

    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        size_t start = t * block;
        size_t end   = (t == T - 1 ? n : start + block);

        uint32_t thread_seed = skip_ahead(seed, start);
        my_lcg gen(thread_seed);

        for (size_t i = start; i < end; ++i) {
            float u = uint32_to_float(gen(), 0.0f, 1.0f);
            result[i] = (u < probability) ? 1.0f : 0.0f;
        }
    }

    return STATUS_OK;
}

