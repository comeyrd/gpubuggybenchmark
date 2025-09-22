#ifndef REFERENCE_ACCURACY_H
#define REFERENCE_ACCURACY_H
#include "accuracy.hpp"

class ReferenceAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif