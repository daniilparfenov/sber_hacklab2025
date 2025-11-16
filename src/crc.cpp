#include <cstddef> 
#include <cstdint>
#include <cmath>
#include <nmmintrin.h>

#include "solution.hpp"

Status crc32(const uint8_t* data, size_t data_len, uint32_t* result) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < data_len; ++i) {
        crc = _mm_crc32_u8(crc, data[i]);
    }
    *result = ~crc;
    return STATUS_OK;
}
