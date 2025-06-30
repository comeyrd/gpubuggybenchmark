
#include <stdio.h>   
#include <stdlib.h> 
#include <chrono>
#include "main.hpp"
#include "fpc.hpp"

int WORK_GROUP_SZ = 200;
int REPEAT = 10;

int main(int argc, char ** argv){
    do_fpc(WORK_GROUP_SZ,REPEAT);
    return 0;
}