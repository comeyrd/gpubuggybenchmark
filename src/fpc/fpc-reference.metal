#include <metal_stdlib>
using namespace metal;

inline uint my_abs(int x) {
    uint t = (uint)(x >> 31);
    return (uint)(x ^ t) - t;
}

inline uint f1(uint64_t value, thread bool* mask) {
    if (value == 0) {
        *mask = true;
    }
    return 1;
}

inline uint f2(uint64_t value, thread bool* mask) {
    if (my_abs((int)(value)) <= 0xFF) *mask = true;
    return 1;
}

inline uint f3(uint64_t value, thread bool* mask) {
    if (my_abs((int)(value)) <= 0xFFFF) *mask = true;
    return 2;
}

inline uint f4(uint64_t value, thread bool* mask) {
    if ((value & 0xFFFF) == 0) *mask = true;
    return 2;
}

inline uint f5(uint64_t value, thread bool* mask) {
    if (my_abs((int)(value & 0xFFFF)) <= 0xFF &&
        my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) {
        *mask = true;
    }
    return 2;
}

inline uint f6(uint64_t value, thread bool* mask) {
    uint byte0 = (uint)(value & 0xFF);
    uint byte1 = (uint)((value >> 8) & 0xFF);
    uint byte2 = (uint)((value >> 16) & 0xFF);
    uint byte3 = (uint)((value >> 24) & 0xFF);
    if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) *mask = true;
    return 1;
}

inline uint f7(uint64_t value, thread bool* mask) {
    *mask = true;
    return 4;
}

kernel void fpc_reference_kernel(const device uint64_t* values [[buffer(0)]],
                                 device atomic_uint* cmp_size [[buffer(1)]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint gid [[thread_position_in_grid]],
                                 uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    threadgroup atomic_uint compressable;
    if (tid == 0) {
        atomic_store_explicit(&compressable, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint64_t value = values[gid];
    uint inc = 0;

    if (value == 0) {
        inc = 1;
    } else if (my_abs((int)(value)) <= 0xFF) {
        inc = 1;
    } else if (my_abs((int)(value)) <= 0xFFFF) {
        inc = 2;
    } else if ((value & 0xFFFF) == 0) {
        inc = 2;
    } else if (my_abs((int)(value & 0xFFFF)) <= 0xFF &&
               my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) {
        inc = 2;
    } else if (((value & 0xFF) == ((value >> 8) & 0xFF)) &&
               ((value & 0xFF) == ((value >> 16) & 0xFF)) &&
               ((value & 0xFF) == ((value >> 24) & 0xFF))) {
        inc = 1;
    } else {
        inc = 4;
    }

    atomic_fetch_add_explicit(&compressable, inc, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == threads_per_threadgroup - 1) {
        uint val = atomic_load_explicit(&compressable, memory_order_relaxed);
        atomic_fetch_add_explicit(cmp_size, val, memory_order_relaxed);
    }
}

kernel void fpc2_reference_kernel(const device uint64_t* values [[buffer(0)]],
                                  device atomic_uint* cmp_size [[buffer(1)]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint gid [[thread_position_in_grid]],
                                  uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    threadgroup atomic_uint compressable;
    if (tid == 0) {
        atomic_store_explicit(&compressable, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool m1 = false, m2 = false, m3 = false, m4 = false, m5 = false, m6 = false, m7 = false;

    uint64_t value = values[gid];
    uint inc1 = f1(value, &m1);
    uint inc2 = f2(value, &m2);
    uint inc3 = f3(value, &m3);
    uint inc4 = f4(value, &m4);
    uint inc5 = f5(value, &m5);
    uint inc6 = f6(value, &m6);
    uint inc7 = f7(value, &m7);

    uint inc = 0;
    if (m1) inc = inc1;
    else if (m2) inc = inc2;
    else if (m3) inc = inc3;
    else if (m4) inc = inc4;
    else if (m5) inc = inc5;
    else if (m6) inc = inc6;
    else inc = inc7;

    atomic_fetch_add_explicit(&compressable, inc, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == threads_per_threadgroup - 1) {
        uint val = atomic_load_explicit(&compressable, memory_order_relaxed);
        atomic_fetch_add_explicit(cmp_size, val, memory_order_relaxed);
    }
}
