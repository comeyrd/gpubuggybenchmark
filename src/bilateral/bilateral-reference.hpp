#ifndef REFERENCE_BILATERAL_H
#define REFERENCE_BILATERAL_H
#include "bilateral.hpp"
#include "cuda-utils.hpp"
class ReferenceBilateral : public IBilateral {
public:
    void setup() override;
    virtual void reset() override {};
    virtual void run(stream_t* s) override;
    virtual void teardown() override;

private:
    float *d_src;
    float *d_dst;
    dim3 threads;
    dim3 blocks;
};

#endif

