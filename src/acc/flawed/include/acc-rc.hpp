#ifndef RC_ACCURACY_H
#define RC_ACCURACY_H
#include "accuracy.hpp"

class RCAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif