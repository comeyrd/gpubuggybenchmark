
#include <stdio.h>   
#include <stdlib.h> 
#include <chrono>

#include "fpc.hpp"

int WORK_GROUP_SZ = 25;
int REPEAT = 5;

int main(int argc, char ** argv){
    do_fpc(WORK_GROUP_SZ,REPEAT);
    return 0;
}