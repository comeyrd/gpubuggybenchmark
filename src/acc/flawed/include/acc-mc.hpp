#ifndef MC_ACCURACY_H
#define MC_ACCURACY_H
#include "accuracy.hpp"

class MCAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif