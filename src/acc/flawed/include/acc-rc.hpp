#ifndef RC_ACCURACY_H
#define RC_ACCURACY_H
#include "accuracy.hpp"

class RCAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif