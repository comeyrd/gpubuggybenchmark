#ifndef CL_ACCURACY_H
#define CL_ACCURACY_H
#include "accuracy.hpp"

class CLAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif