#include <metal_stdlib>
using namespace metal;

inline int my_abs(int x) {
    int t = (int)(x >> 31);
    return (int)(x ^ t) - t;
}

inline int f1(uint64_t value, thread bool* mask) {
    if (value == 0) {
        *mask = true;
    }
    return 1;
}

inline int f2(uint64_t value, thread bool* mask) {
    if (my_abs((int)(value)) <= 0xFF) *mask = true;
    return 1;
}

inline int f3(uint64_t value, thread bool* mask) {
    if (my_abs((int)(value)) <= 0xFFFF) *mask = true;
    return 2;
}

inline int f4(uint64_t value, thread bool* mask) {
    if ((value & 0xFFFF) == 0) *mask = true;
    return 2;
}

inline int f5(uint64_t value, thread bool* mask) {
    if (my_abs((int)(value & 0xFFFF)) <= 0xFF &&
        my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) {
        *mask = true;
    }
    return 2;
}

inline int f6(uint64_t value, thread bool* mask) {
    int byte0 = (int)(value & 0xFF);
    int byte1 = (int)((value >> 8) & 0xFF);
    int byte2 = (int)((value >> 16) & 0xFF);
    int byte3 = (int)((value >> 24) & 0xFF);
    if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) *mask = true;
    return 1;
}

inline int f7(uint64_t value, thread bool* mask) {
    *mask = true;
    return 4;
}

kernel void fpc_reference_kernel(const device uint64_t* values [[buffer(0)]],
                                 device atomic_uint* cmp_size [[buffer(1)]],
                                 int tid [[thread_position_in_threadgroup]],
                                 int gid [[thread_position_in_grid]],
                                 int threads_per_threadgroup [[threads_per_threadgroup]])
{
    threadgroup atomic_uint compressable;
    if (tid == 0) {
        atomic_store_explicit(&compressable, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint64_t value = values[gid];
    int inc = 0;

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
        int val = atomic_load_explicit(&compressable, memory_order_relaxed);
        atomic_fetch_add_explicit(cmp_size, val, memory_order_relaxed);
    }
}

kernel void fpc2_reference_kernel(const device uint64_t* values [[buffer(0)]],
                                  device atomic_uint* cmp_size [[buffer(1)]],
                                  int tid [[thread_position_in_threadgroup]],
                                  int gid [[thread_position_in_grid]],
                                  int threads_per_threadgroup [[threads_per_threadgroup]])
{
    threadgroup atomic_uint compressable;
    if (tid == 0) {
        atomic_store_explicit(&compressable, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool m1 = false, m2 = false, m3 = false, m4 = false, m5 = false, m6 = false, m7 = false;

    uint64_t value = values[gid];
    int inc1 = f1(value, &m1);
    int inc2 = f2(value, &m2);
    int inc3 = f3(value, &m3);
    int inc4 = f4(value, &m4);
    int inc5 = f5(value, &m5);
    int inc6 = f6(value, &m6);
    int inc7 = f7(value, &m7);

    int inc = 0;
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
        int val = atomic_load_explicit(&compressable, memory_order_relaxed);
        atomic_fetch_add_explicit(cmp_size, val, memory_order_relaxed);
    }
}
