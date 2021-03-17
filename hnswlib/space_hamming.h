#pragma once
#include "hnswlib.h"
#include <stdlib.h>

extern const uint8_t lookup8bit[256];

namespace hnswlib {

#define fast_loop_imp(fun_u64, fun_u8) \
    auto a = reinterpret_cast<const uint64_t*>(data1); \
    auto b = reinterpret_cast<const uint64_t*>(data2); \
    int div = code_size / 8; \
    int mod = code_size % 8; \
    int i = 0, len = div; \
    switch(len & 7) { \
        default: \
            while (len > 7) { \
                len -= 8; \
                fun_u64; i++; \
                case 7: fun_u64; i++; \
                case 6: fun_u64; i++; \
                case 5: fun_u64; i++; \
                case 4: fun_u64; i++; \
                case 3: fun_u64; i++; \
                case 2: fun_u64; i++; \
                case 1: fun_u64; i++; \
            } \
    } \
    if (mod) { \
        auto a = data1 + 8 * div; \
        auto b = data2 + 8 * div; \
        switch (mod) { \
            case 7: fun_u8(6); \
            case 6: fun_u8(5); \
            case 5: fun_u8(4); \
            case 4: fun_u8(3); \
            case 3: fun_u8(2); \
            case 2: fun_u8(1); \
            case 1: fun_u8(0); \
            default: break; \
        } \
    }

int xor_popcnt(const uint8_t* data1, const uint8_t*data2, const size_t code_size) {
#define fun_u64 accu += __builtin_popcountl(a[i] ^ b[i]);
#define fun_u8(i) accu += lookup8bit[a[i] ^ b[i]];
    int accu = 0;
    fast_loop_imp(fun_u64, fun_u8);
    return accu;
#undef fun_u64
#undef fun_u8
}

int hamming(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return xor_popcnt((const uint8_t*)pVect1, (const uint8_t*)pVect2, *(size_t*)qty_ptr);
}

class HammingSpace : public SpaceInterface<int> {

DISTFUNC<int> fstdistfunc_;
size_t data_size_;
size_t dim_;
public:
HammingSpace(size_t dim) {
    fstdistfunc_ = hamming;
    dim_ = dim;
    data_size_ = dim / 8;
}

size_t get_data_size() {
    return data_size_;
}

DISTFUNC<int> get_dist_func() {
    return fstdistfunc_;
}

void *get_dist_func_param() {
    return &data_size_;
}

~HammingSpace() {}
};

}
