#ifndef ML_ACCURACY_H
#define ML_ACCURACY_H
#include "accuracy.hpp"

class MLAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif