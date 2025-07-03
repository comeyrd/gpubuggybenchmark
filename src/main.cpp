
#include <stdio.h>   
#include <stdlib.h> 
#include <chrono>
#include "main.hpp"


int main(int argc, char ** argv){
    argparse::ArgumentParser program("gb-benchmark", VERSION_STRING);
    program.add_argument("--list-versions").flag();

    class_umap<IKernel> kernels = Manager<IKernel>::instance()->getClasses();

    for(const auto & [name,func]:kernels){
        std::string versions = func->list_versions();
        std::cout
    }
    return 0;
}