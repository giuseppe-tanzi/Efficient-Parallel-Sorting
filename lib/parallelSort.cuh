#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/utils.cuh"

// Define a structure to hold block and thread configurations
struct ParallelSortConfig
{
    dim3 gridSize;
    dim3 blockSize;
    unsigned long partitionSize;
    unsigned long nTotalThreads;
    unsigned long nBlocks;
    unsigned long nThreadsPerBlock;
};

ParallelSortConfig determine_config(const unsigned long N);