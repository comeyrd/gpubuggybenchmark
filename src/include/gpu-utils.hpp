#ifndef GPUUTILS_HPP
#define GPUUTILS_HPP
void setup_gpu();
void reset_gpu();
struct stream_t;
struct event_t;

class GpuStream{
    public:
        GpuStream();
        ~GpuStream();
        stream_t* get_stream() const {
            return stream;
        }
        bool get_stream_availability();
        void synchronize();
    private:
        stream_t* stream;
};
#include "KernelStats.hpp"

#include "Types.hpp"

struct l2flushr {
public:
    l2flushr();
    ~l2flushr();
    void flush(stream_t* stream);

private:
    stream_t *cs;
    int buffer_size{};
    int *l2_buffer{};
};

struct blocking_kernel {
    blocking_kernel();
    ~blocking_kernel();

    void block(stream_t *stream, double timeout);

    inline void unblock() {
        volatile int32_t &flag = m_host_flag;
        flag = 1;

        const volatile int32_t &timeout_flag = m_host_timeout_flag;
        if (timeout_flag) {
            blocking_kernel::timeout_detected();
        }
    }

    // move-only
    blocking_kernel(const blocking_kernel &) = delete;
    blocking_kernel(blocking_kernel &&) = default;
    blocking_kernel &operator=(const blocking_kernel &) = delete;
    blocking_kernel &operator=(blocking_kernel &&) = default;

private:
    int32_t m_host_flag{};
    int32_t m_host_timeout_flag{};
    int32_t *m_device_flag{};
    int32_t *m_device_timeout_flag{};

    static void timeout_detected();
};

class GpuEventTimer {
private:
    event_t *memstart2D;
    event_t *memstop2D;
    event_t *memstart2H;
    event_t *memstop2H;
    event_t **warmupstart;
    event_t **warmupstop;
    event_t **repetitionstart;
    event_t **repetitionstop;
    stream_t *stream;
    int nb_w = 0; // what warmup iteration are we on
    int nb_r = 0; // what repetition iteration are we on
    int m_warmup;
    int m_repetitions;
public:
    GpuEventTimer(const int& warmup, const int& repetitions, stream_t *gpustream);
    ~GpuEventTimer();
    void begin_mem2D();
    void end_mem2D();
    void begin_mem2H();
    void end_mem2H();
    void begin_warmup();
    void end_warmup();
    void begin_repetition();
    void end_repetition();
    KernelStats retreive();
};

#endif