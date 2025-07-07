#ifndef ACCURACY_H
#define ACCURACY_H
#include <memory>
#include <string>
#include <unordered_map>
#include "Manager.hpp"
#include "Kernel.hpp"
#include "version.hpp"

class IAccuracy {
public:
    virtual ~IAccuracy() = default;
    virtual void accuracy(int nrows, int ndims, int top_k, int repeat, int* label, float *data) const = 0;
};

class Accuracy : public IKernel{
    public:
        int run_kernel(int argc,char** argv);
    private : 
        void run_versions(class_umap<IFpc> versions,int repetitions);
        void register_cli_options(argparse::ArgumentParser& parser) override;
        std::vector<std::string> list_versions() override;
};



#endif