#ifndef Kernel_HPP
#define Kernel_HPP
#include <argparse/argparse.hpp>

class IKernel{
    public:
        virtual ~IKernel() = default;
        virtual void register_cli_options(argparse::ArgumentParser& parser) = 0;
        virtual std::vector<std::string> list_versions() = 0;
        virtual void run_versions(std::vector<std::string> versions, const argparse::ArgumentParser& parsed_args) = 0;
};

#endif
