#ifndef REFERENCE_ACCURACY_H
#define REFERENCE_ACCURACY_H
#include "accuracy.hpp"

class ReferenceAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif