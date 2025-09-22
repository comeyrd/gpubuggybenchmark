#ifndef MAIN_H
#define MAIN_H
#include "version.hpp"
#include "Manager.hpp"
#include "Kernel.hpp"
#include <argparse/argparse.hpp>
void list_kernels();
void run_all(int argc, char** argv);
kernel_pair get_kernel(std::string kernel);
#endif
