
#include "main.hpp"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept> 

int main(int argc, char **argv) {
    argparse::ArgumentParser program("gb-benchmark", VERSION_STRING,argparse::default_arguments::version);
    bool list;
    bool help;
    bool all;
    std::string kernel;
    auto &group = program.add_mutually_exclusive_group(true);
    group.add_argument("kernel").nargs(argparse::nargs_pattern::optional).store_into(kernel).help("Name of the selected Kernel");
    group.add_argument("-lk", "--list-kernels").help("lists available kernels").store_into(list);
    group.add_argument("-A").help("Run all kernels and all versions").store_into(all);
    group.add_argument("-h").help("shows help message and exits").store_into(help);
    program.add_argument("--repetitions", "-r").default_value(400).scan<'i', int>().help("Number of repetitions of execution of each Version");


    try {
        program.parse_known_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    if (list) {
        list_kernels();
    }else if(all){
        run_all(argc,argv);
    } else if(program.is_used("kernel")) {
        try{
            kernel_pair kp = get_kernel(kernel);
            kp.second->run(argc,argv);
        }catch(const std::exception &e){
            std::cerr << e.what() << std::endl;
            return 1;
        }
    }else if(help){
        std::cerr << program;
        return 0;
    }
    else{
        std::cerr << "Missing 'kernel' or '--list-kernels' argument"<<std::endl;
        std::cerr << program;
        return 1;
    }
    return 0;
}

void list_kernels() {
    class_umap<I_IKernel> kernels = Manager<I_IKernel>::instance()->getClasses();
    std::cout << "Available Kernels :  "<<std::endl;
    for (const kernel_pair &pair : kernels) {
        std::cout << pair.first << std::endl;
    }
}

void run_all(int argc, char** argv){
    class_umap<I_IKernel> kernels = Manager<I_IKernel>::instance()->getClasses();
     for (const kernel_pair &pair : kernels) {
        pair.second->run(argc,argv);
    }
}

kernel_pair get_kernel(std::string kernel){
    class_umap<I_IKernel> kernels = Manager<I_IKernel>::instance()->getClasses();
    for (const kernel_pair &pair : kernels) {
        const std::string &name = pair.first;
        if(name.compare(kernel) == 0)
            return pair;
    }
    throw std::runtime_error(std::string("Kernel ") + kernel + " not found");
}

