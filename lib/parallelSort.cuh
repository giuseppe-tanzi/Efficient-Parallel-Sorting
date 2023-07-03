#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/mergeSort.cuh"
#include "../lib/radixSort.cuh"
#include "../lib/utils.cuh"

// Structure to hold block and thread configurations
struct ParallelSortConfig
{
    dim3 gridSize;
    dim3 blockSize;
    unsigned long long partitionSize;
    unsigned long nTotalThreads;
    unsigned long nBlocks;
    unsigned long nThreadsPerBlock;
};

/*
    Ensures the minimum numbers of necessary thread
        - First branch: N is smaller or equal than the starting partition size of each thread
            - Starting from the maximum number of thread needed (N), it checks that the number of threads is a power of two,
                otherwise the merging phase will not work
        - Second branch: N is greater than the starting partition size of each thread
            - It checks that the number of necessary threads is smaller or equal than the number of threads for each block
                and it computes the partition size
            - If the number of necessary threads is smaller than the number of threads for each blocks,
                it does the same thing of the first branch starting from the number of necessary thread
            - Otherwise it computes the number of minimum blocks needed ensuring that the number of threads is a power of 2

*/
ParallelSortConfig determine_config(const unsigned long long N);

/*
    Entire sort kernel:
        1. Radix sort
        2. Merge sort
*/
__global__ void sort_kernel(unsigned short *data, unsigned long long n, unsigned long long offset, const unsigned long n_threads);