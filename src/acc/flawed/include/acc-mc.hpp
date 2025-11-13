#ifndef MC_ACCURACY_H
#define MC_ACCURACY_H
#include "accuracy.hpp"

class MCAccuracy : public IAccuracy {
public:
    void setup() override;
    void reset() override;
    void run(stream_t* s) override;
    void teardown(AccuracyResult &_result) override;
private:
    int *d_label;
    float *d_data;
    int *d_count;
    dim3 block;
    dim3 grid;
};

#endif
