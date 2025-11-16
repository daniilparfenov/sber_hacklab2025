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

    for (size_t i = 0; i < n; i += 2) {
        // Генерация двух uniform чисел [0,1)
        float u1 = uint32_to_float(gen(), 0.0f, 1.0f);
        float u2 = uint32_to_float(gen(), 0.0f, 1.0f);

        // Box-Muller преобразование
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * 3.14159265358979323846f * u2;

        // Получаем два нормальных числа
        result[i] = mean + stddev * r * cosf(theta);
        if (i + 1 < n)
            result[i + 1] = mean + stddev * r * sinf(theta);
    }

    return STATUS_OK;
}

Status generate_exponential(size_t n, uint32_t seed, float lambda, float* result) {
    my_lcg gen(seed);

    for (size_t i = 0; i < n; ++i) {
        result[i] = -logf(1.0f - uint32_to_float(gen(), 0.f, 1.f)) / lambda;
    }

    return STATUS_OK;
}

Status generate_bernoulli(size_t n, uint32_t seed, float probability, float* result) {
    my_lcg gen(seed);

    for (size_t i = 0; i < n; ++i) {
        result[i] = (uint32_to_float(gen(), 0.0f, 1.0f) < probability) ? 1.0f : 0.0f;
    }

    return STATUS_OK;
}
