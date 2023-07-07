#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/utils.cuh"
#include "../lib/utilsParallelSort.cuh"

/* 
    Merge function of the MergeSort Algorithm 
*/
void merge(unsigned short *data, const unsigned long long left, const unsigned long long mid, const unsigned long long right);

/* 
    Merge function of the MergeSort Algorithm to run on GPU
*/
__device__ void merge_gpu(unsigned short *data, const unsigned long long left, const unsigned long long mid, const unsigned long long right);

/* 
    Merge Sort Algorithm 
*/
__host__ void merge_sort(unsigned short *data, const unsigned long long left, const unsigned long long right);

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