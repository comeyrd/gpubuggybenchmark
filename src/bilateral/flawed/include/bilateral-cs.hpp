#ifndef CS_BILATERAL_H
#define CS_BILATERAL_H
#include "bilateral.hpp"

class CSBilateral : public IBilateral {
public:
    KernelStats run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const override;
};

#endif