#ifndef FPCREFERENCE
#define FPCREFERENCE
#include "fpc.hpp"

class ReferenceFPC : public IFPC {
public:
    void setup() override;
    void reset() override;
    void run(stream_t* s) override;
    void teardown(FPCResult& result) override;
private:
    ulong* d_values;
    unsigned* d_cmp_size;
    size_t* d_length;
    dim3 grids;
    dim3 threads;
};

class ReferenceFPC2 : public IFPC {
public:
    void setup() override;
    void reset() override;
    void run(stream_t* s) override;
    void teardown(FPCResult& result) override;
private:
    ulong* d_values;
    unsigned* d_cmp_size;
    size_t* d_length;
    dim3 grids;
    dim3 threads;
};

#endif // FPCREFERENCE
