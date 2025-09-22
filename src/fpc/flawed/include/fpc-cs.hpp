#ifndef FPCCS
#define FPCCS
#include "fpc.hpp"

class CSFPC : public IFPC {
public:
    KernelStats run(const FPCData &data, const FPCSettings &settings, FPCResult &result) const override;
};


#endif