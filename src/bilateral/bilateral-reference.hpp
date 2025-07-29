#ifndef REFERENCE_BILATERAL_H
#define REFERENCE_BILATERAL_H
#include "bilateral.hpp"

class ReferenceBilateral : public IBilateral {
public:
    KernelStats bilateral(const BilateralData &aData, const BilateralSettings &aSettings, BilateralResult &aResult) const override;
};

#endif