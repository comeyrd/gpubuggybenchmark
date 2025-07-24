#ifndef DL_ACCURACY_H
#define DL_ACCURACY_H
#include "accuracy.hpp"

class DLAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif