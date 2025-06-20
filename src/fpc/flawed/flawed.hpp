#ifndef FLAWED
#define FLAWED
#include "fpc.hpp"

#include "fpc-ml.hpp"
#include "fpc-rc.hpp"
#include "fpc-reference.hpp"

Kernel_umap retrieve_kernels(){
    Kernel_umap kmap;
    kmap["RF"] = std::make_shared<ReferenceFpc>();
    kmap["ML"] = std::make_shared<MLFpc>();
    kmap["RC"] = std::make_shared<RCFpc>();
    return kmap;
}

#endif