#ifndef FPC_H
#define FPC_H
#include <memory>
#include <string>
#include <unordered_map>
#include "Manager.hpp"
#include "Kernel.hpp"
#include "version.hpp"

typedef unsigned long ulong;
constexpr int WORK_GROUP_SZ = 200;

class IFpc {
public:
    virtual ~IFpc() = default;
    virtual void fpc(const ulong *values, unsigned *cmp_size_hw, const int values_size, const int wgs) const = 0;
    virtual void fpc2(const ulong *values, unsigned *cmp_size_hw, const int values_size, const int wgs) const = 0;
};

ulong *convertBuffer2Array(char *cbuffer, unsigned size, unsigned step);


class FPC : public IKernel{
    public:
        int run_kernel(int argc,char** argv);
    private : 
        void run_versions(class_umap<IFpc> versions,int repetitions);
        void run_impl(std::shared_ptr<IFpc> fpc_impl, ulong* values, unsigned values_size, int cmp_size, int work_group_sz, int repeat);
        void register_cli_options(argparse::ArgumentParser& parser) override;
        std::vector<std::string> list_versions() override;
};


#endif
