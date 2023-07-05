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
    unsigned long long partition_size;
    unsigned long total_threads;
    unsigned long total_blocks;
    unsigned long threads_per_block;
};

/*
    Entire sort kernel:
        1. Radix sort
        2. Merge sort
*/
__global__ void sort_kernel(unsigned short *data, unsigned long long n, unsigned long long offset, const unsigned long n_threads);

/*
    The merge_kernel function is a CUDA kernel that performs a merging phase of the merge sort on data in parallel using total threads on one block
    It divides the array into smaller ranges and merges them progressively until the entire array is sorted.
*/
__global__ void merge_kernel(unsigned short *data, const unsigned long *offset, const unsigned long total_threads, const unsigned long total_threads_precedent_blocks);