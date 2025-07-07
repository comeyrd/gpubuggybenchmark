#include "fpc-reference.hpp"
#include <stdio.h>   
#include <stdlib.h> 
#include <chrono>
#include <memory>
#include <iostream>
#include "fpc.hpp"

ulong* convertBuffer2Array (char *cbuffer, unsigned size, unsigned step)
{
  unsigned i,j; 
  ulong * values = NULL;
  posix_memalign((void**)&values, 1024, sizeof(ulong)*size/step);
  for (i = 0; i < size / step; i++) {
    values[i] = 0;    // Initialize all elements to zero.
  }
  for (i = 0; i < size; i += step ){
    for (j = 0; j < step; j++){
      values[i / step] += (ulong)((unsigned char)cbuffer[i + j]) << (8*j);
    }
  }
  return values;
}


void run_fpc_impl(std::shared_ptr<IFpc> fpc_impl, ulong* values, unsigned values_size, int cmp_size, int work_group_sz, int repeat){

// run on the device
  unsigned cmp_size_hw; 

  bool ok = true;
  // warmup
  fpc_impl->fpc(values, &cmp_size_hw, values_size, work_group_sz);

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    fpc_impl->fpc(values, &cmp_size_hw, values_size, work_group_sz);
    if (cmp_size_hw != cmp_size) {
      printf("fpc failed %u != %u\n", cmp_size_hw, cmp_size);
      ok = false;
      break;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("fpc: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

  // warmup
  fpc_impl->fpc2(values, &cmp_size_hw, values_size, work_group_sz);

  start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    fpc_impl->fpc2(values, &cmp_size_hw, values_size, work_group_sz);
    if (cmp_size_hw != cmp_size) {
      printf("fpc2 failed %u != %u\n", cmp_size_hw, cmp_size);
      ok = false;
      break;
    }
  }

  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("fpc2: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

  printf("%s\n", ok ? "PASS" : "FAIL");

}

void FPC::register_cli_options(argparse::ArgumentParser& parser) {
  auto &group = parser.add_mutually_exclusive_group(true);
  group.add_argument("-a", "--all").flag();
  group.add_argument("-v", "--version").nargs(argparse::nargs_pattern::at_least_one);
  group.add_argument("-lv","--list-versions").flag();
  parser.add_argument("--repetitions", "-r").default_value(5);
};


int FPC::run_kernel(int argc, char** argv){
   if(argc < 2){
    std::cerr << "Not enough arguments"<<std::endl;
    return 1;
  }
  std::string name = std::string(argv[0]) + " " + argv[1];
  argparse::ArgumentParser fpc_parser(name,VERSION_STRING);
  this->register_cli_options(fpc_parser);
  
    try {
        fpc_parser.parse_known_args(argc,argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << fpc_parser;
        return 1;
    }

  bool all_set = fpc_parser.get<bool>("--all");
  bool list_set = fpc_parser.get<bool>("--list-versions");
  int repetitions = fpc_parser.get<int>("--repetitions");
  bool versions_set = fpc_parser.is_used("-v");
  

  if(list_set){
    std::cout << "Versions of FPC :" << std::endl;
    for(const auto& version : this->list_versions()){
      std::cout << version << std::endl;
    }
  }else if(all_set){
      class_umap<IFpc> versions_map = Manager<IFpc>::instance()->getClasses();
      this->run_versions(versions_map,repetitions);
  }else if(versions_set){
    std::vector<std::string> versions =  fpc_parser.get<std::vector<std::string>>("-v");
    class_umap<IFpc> versions_map = select_versions_in_umap(versions,Manager<IFpc>::instance()->getClasses());
    this->run_versions(versions_map,repetitions);
  }else{
    std::cout <<fpc_parser<<std::endl;
    return 1;
  }
  return 0;
}

std::vector<std::string> FPC::list_versions(){
  class_umap<IFpc> versions = Manager<IFpc>::instance()->getClasses();
  std::vector<std::string> vs;
  for(const auto &[name, _ ] : versions){
    vs.push_back(name);
  }
  return vs;
}

void FPC::run_versions(class_umap<IFpc> versions,int repetitions){

  const int step = 4;
  const size_t size = (size_t)WORK_GROUP_SZ * WORK_GROUP_SZ * WORK_GROUP_SZ;
  char* cbuffer = (char*) malloc (size * step);

  srand(2);
  for (size_t i = 0; i < size*step; i++) {
    cbuffer[i] = 0xFF << (rand() % 256);
  }

  ulong *values = convertBuffer2Array (cbuffer, size, step);
  unsigned values_size = size / step;

  unsigned cmp_size = fpc_cpu(values, values_size);

  for (const auto &[name, k_func] : versions) {
    std::cout <<" Kernel "<< name << std::endl;
    run_fpc_impl(k_func,values,values_size,cmp_size, WORK_GROUP_SZ, repetitions);
    std::cout << std::endl;
  }

  free(values);
  free(cbuffer);
}

REGISTER_CLASS(IKernel,FPC)