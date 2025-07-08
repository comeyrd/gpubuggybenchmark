#ifndef Kernel_HPP
#define Kernel_HPP
#include <argparse/argparse.hpp>
#include "Manager.hpp"

class IKernel{
    public:
        virtual ~IKernel() = default;
        virtual int run_kernel(int argc,char** argv) = 0;
    private: 
        virtual void register_cli_options(argparse::ArgumentParser& parser) = 0;
        virtual std::vector<std::string> list_versions() = 0;
};

typedef class_pair<IKernel> kernel_pair ;


#endif
