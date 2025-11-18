#ifndef DG_BILATERAL_H
#define DG_BILATERAL_H
#include "bilateral.hpp"

class DGBilateral : public IBilateral {
public:
    void setup() override;
    virtual void reset() override {};
    virtual void run(stream_t* s) override;
    virtual void teardown(BilateralData &_result) override;

private:
    float *d_src;
    float *d_dst;
    dim3 threads;
    dim3 blocks;
};

#endif