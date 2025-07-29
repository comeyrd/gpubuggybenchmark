# GPU Buggy Benchmark

For this benchmark, we implement bugs into kernels to test if performance analysis tools for GPU programs are able to detect them.

First we work on the following source kernels : 

FPC from the HeCBench, but originally from here : 

FPDC from the HeCBench, but originally from here : [GFC](https://userweb.cs.txstate.edu/~burtscher/research/GFC/)

LZSS from the HeCBench, but originally from here : [GPULZ](https://github.com/hpdps-group/ICS23-GPULZ)

Accuracy from the HeCBench, but originally from the PyTorch Library

Bilateral from the HeCBench, but originally from [Here](https://github.com/jstraub/cudaPcl)

Support for METAL and mac works if the metal-cpp headers are in path : [metal-cpp](https://developer.apple.com/metal/cpp/)

