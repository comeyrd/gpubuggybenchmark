#ifndef LU_BILATERAL_H
#define LU_BILATERAL_H
#include "bilateral.hpp"

class LUBilateral : public IBilateral {
public:
    KernelStats run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const override;
};

#endif