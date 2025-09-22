#ifndef AB_BILATERAL_H
#define AB_BILATERAL_H
#include "bilateral.hpp"

class ABBilateral : public IBilateral {
public:
    KernelStats run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const override;
};

#endif