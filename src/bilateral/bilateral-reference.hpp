#ifndef REFERENCE_BILATERAL_H
#define REFERENCE_BILATERAL_H
#include "bilateral.hpp"

class ReferenceBilateral : public IBilateral {
public:
    KernelStats run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const override;
};

#endif