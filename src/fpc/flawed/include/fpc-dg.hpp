#ifndef FPCDG
#define FPCDG
#include "fpc.hpp"

class DGFPC : public IFPC {
public:
    KernelStats run(const FPCData &data, const FPCSettings &settings, FPCResult &result) const override;
};


#endif