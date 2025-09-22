#ifndef UO_ACCURACY_H
#define UO_ACCURACY_H
#include "accuracy.hpp"

class UOAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif