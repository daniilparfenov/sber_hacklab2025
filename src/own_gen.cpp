#include <cstdint>

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
        // Используем 64-битное умножение для безопасного переполнения
        uint64_t product = uint64_t(state) * multiplier;
        // Оптимизация: заменяем модуль через битовые операции
        state = (product >> 31) + (product & modulus);
        if (state > modulus) state -= modulus;
        return state;
    }
};
