#ifndef CO_BILATERAL_H
#define CO_BILATERAL_H
#include "bilateral.hpp"

class COBilateral : public IBilateral {
public:
    KernelStats run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const override;
};

#endif