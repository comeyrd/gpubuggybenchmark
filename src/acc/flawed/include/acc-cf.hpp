#ifndef CF_ACCURACY_H
#define CF_ACCURACY_H
#include "accuracy.hpp"

class CFAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif