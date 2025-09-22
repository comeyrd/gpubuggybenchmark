#ifndef FPCCF
#define FPCCF
#include "fpc.hpp"

class CFFPC : public IFPC {
public:
    KernelStats run(const FPCData &data, const FPCSettings &settings, FPCResult &result) const override;
};


#endif