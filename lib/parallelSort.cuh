#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/mergeSort.cuh"
#include "../lib/radixSort.cuh"
#include "../lib/utils.cuh"
#include "../lib/utilsParallelSort.cuh"

/*
    Function that performs parallel sorting
    There are two different ways to compute parallel sorting based on the number of blocks:
        - One way is to compute the radix sort phase and the merging sort phase in the same kernel.
        - The second way is to compute the sorting in three different phases:
            - First step:
                - Radix sort on the entire array with all the necessary blocks.
            - Second step:
                - Merge phase, assigning a different partition of the array to each block.
            - Third step:
                - Merge block phase, merging all the sorted partitions of the array sorted by each block.
*/
void parallel_sort(unsigned short *dev_a, const unsigned long long N,
                   ParallelSortConfig config,
                   const size_t size_blocks,
                   const unsigned blocks_involved_in_merging,
                   unsigned long long *block_starting_idx,
                   unsigned long long *block_size,
                   unsigned long *thread_offset,
                   unsigned long *dev_thread_offset);

/*
    Entire sort kernel:
        1. Radix sort
        2. Merge sort
*/
__global__ void sort_kernel(unsigned short *data, const unsigned long long N, unsigned long long offset, const unsigned long n_threads);

/*
    - The merge_kernel function is a CUDA kernel that performs a merging phase of the merge sort on data in parallel using total threads on one block
    - It divides the array into smaller ranges and merges them progressively until the entire array is sorted.
*/
__global__ void merge_kernel(unsigned short *data, const unsigned long *offset, const unsigned long total_threads, const unsigned long total_threads_precedent_blocks);

/*
    - The merge_block_kernel is a CUDA kernel for merging sorted array on one single block. 
    - It performs a parallel merge sort on the data array using multiple threads on one single block.
    - The function merges the array in a hierarchical manner until the entire data array is sorted.
*/
__global__ void merge_blocks_kernel(unsigned short *data, unsigned long long N, ParallelSortConfig config, const unsigned total_threads);