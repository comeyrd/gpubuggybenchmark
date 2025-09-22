#ifndef IL_BILATERAL_H
#define IL_BILATERAL_H
#include "bilateral.hpp"

class ILBilateral : public IBilateral {
public:
    KernelStats run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const override;
};

#endif