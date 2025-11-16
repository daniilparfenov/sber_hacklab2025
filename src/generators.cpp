#include <cstddef> 
#include <cstdint>
#include <random>
#include <omp.h>
#include <algorithm>
#include <immintrin.h>

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
        size_t end   = (t == (int)(T-1) ? n : start + block);

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
        size_t end = (t == (int)(T-1) ? n : start + block);

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
        size_t end   = (t == (int)(T-1) ? n : start + block);

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

struct xoshiro128pp {
    uint32_t s[4];

    xoshiro128pp(uint32_t seed) {
        uint32_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9E3779B9;
            uint32_t t = z;
            t ^= t >> 16;
            t *= 0x85EBCA6B;
            t ^= t >> 13;
            t *= 0xC2B2AE35;
            t ^= t >> 16;
            s[i] = t;
        }
    }

    static inline uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

    uint32_t operator()() {
        uint32_t result = rotl(s[0] + s[3], 7) + s[0];
        uint32_t t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 11);
        return result;
    }
};

// Convert 32-bit integer to float in [0,1)
inline float uint32_to_float(uint32_t x) {
    return (x >> 9) * (1.0f / 8388608.0f); // 23-bit mantissa
}

// -----------------------------------------
// Vectorized log approximation (8 floats, AVX2)
inline __m256 log_ps(__m256 x) {
    // Using a simple natural log approximation
    // Reference: https://stackoverflow.com/questions/39821367/fast-log2float-in-c
    alignas(32) float xf[8]; _mm256_store_ps(xf, x);
    for(int i = 0; i < 8; ++i) xf[i] = std::log(xf[i]);
    return _mm256_load_ps(xf);
}

// -----------------------------------------
// Exponential RNG using OpenMP + SIMD
Status generate_exponential(size_t n, uint32_t seed, float lambda, float* result) {
    size_t T = omp_get_max_threads();
    size_t block = (n + T - 1) / T;

#pragma omp parallel
    {
        int t = omp_get_thread_num();
        size_t start = t * block;
        size_t end = std::min(start + block, n);

        xoshiro128pp gen(seed + t); // per-thread seed

        float inv_lambda = 1.0f / lambda;

        size_t i = start;

        // Vectorized 8 elements at a time
        for (; i + 7 < end; i += 8) {
            alignas(32) float u_arr[8];
            for (int j = 0; j < 8; ++j)
                u_arr[j] = uint32_to_float(gen());

            __m256 u = _mm256_load_ps(u_arr);
            __m256 logu = log_ps(u);       // approximate log(u)
            __m256 exp_val = _mm256_mul_ps(_mm256_set1_ps(-inv_lambda), logu);
            _mm256_storeu_ps(result + i, exp_val);
        }

        // Tail: remaining elements
        for (; i < end; ++i) {
            float u = uint32_to_float(gen());
            result[i] = -logf(u) * inv_lambda;
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
        size_t end   = (t == (int)(T-1) ? n : start + block);

        uint32_t thread_seed = skip_ahead(seed, start);
        my_lcg gen(thread_seed);

        for (size_t i = start; i < end; ++i) {
            float u = uint32_to_float(gen(), 0.0f, 1.0f);
            result[i] = (u < probability) ? 1.0f : 0.0f;
        }
    }

    return STATUS_OK;
}

