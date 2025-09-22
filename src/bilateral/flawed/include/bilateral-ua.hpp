#ifndef UA_BILATERAL_H
#define UA_BILATERAL_H
#include "bilateral.hpp"

class UABilateral : public IBilateral {
public:
    KernelStats run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const override;
};

#endif