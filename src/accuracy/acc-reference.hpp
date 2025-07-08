#ifndef REFERENCE_ACCURACY_H
#define REFERENCE_ACCURACY_H
#include "accuracy.hpp"

class ReferenceAccuracy : public IAccuracy {
public:
    int accuracy(const AccuracyData &aData, const AccuracySettings &aSettings) const override;
};

#endif