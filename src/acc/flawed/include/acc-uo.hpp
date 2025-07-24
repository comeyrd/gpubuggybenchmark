#ifndef UO_ACCURACY_H
#define UO_ACCURACY_H
#include "accuracy.hpp"

class UOAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif