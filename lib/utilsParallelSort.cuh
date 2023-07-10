#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
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
    size_t required_shared_memory;
    int max_shared_memory_per_block;
};

/*
    Function that determines the configuration parameters for parallel sorting based on the input size N. 
    It calculates the number of threads, blocks, partition size, and other parameters required for efficient parallel sorting.
*/
ParallelSortConfig determine_config(const unsigned long long N);

/*
    Function that returns the number of lists to merge at level 0 of the merging phase
*/
unsigned long get_n_list_to_merge(unsigned long long N, unsigned long long partition, unsigned long total_threads);

/*
    Function that compute the starting index of data blocks for the parallel computation,
    given a set of parameters:
        - block_starting_idx: A pointer to an array representing the block starting idx. The starting index of each block will be stored in this array.
        - N: An unsigned long long integer representing the total number of elements in the data.
        - total_blocks: An unsigned integer representing the total number of blocks.
        - threads_per_block: An unsigned long integer representing the total number of threads per block.
        - total_threads: An unsigned long integer representing the total number of threads.
    It modifies the array block_starting_idx.
    It follows a recursive approach to distribute the workload evenly among the threads and blocks.
*/
__host__ __device__ void get_start_index_block(unsigned long long *block_starting_idx, const unsigned long long N, const unsigned total_blocks, const unsigned long threads_per_block, const unsigned long total_threads);

/*
    Function that compute the size of data blocks for the parallel computation,
    given a set of parameters:
        - block_size: A pointer to an array representing the size of the array assigned to each block. The size of each block will be stored in this array.
        - block_starting_idx: A pointer to an array representing the block starting idx.
        - N: An unsigned long long integer representing the total number of elements in the data.
        - total_blocks: An unsigned integer representing the total number of blocks.
        - threads_per_block: An unsigned long integer representing the total number of threads per block.
        - total_threads: An unsigned long integer representing the total number of threads.
    It modifies the array block_size.
    It follows a recursive approach to distribute the workload evenly among the threads and blocks.
*/
__host__ __device__ void get_size_block(unsigned long long *block_size, const unsigned long long *block_starting_idx, const unsigned long long N, const unsigned total_blocks, const unsigned long threads_per_block, const unsigned long total_threads);

/*
    Function that compute the offset for each thread of each block needed for the merging phase during the parallel computation,
    given a set of parameters:
        - offsets: A pointer to an array representing the offsets for each thread in each block. The offset for each thread will be accumulated in this array.
        - block_starting_idx: A pointer to an array representing the block starting idx.
        - N: An unsigned long long integer representing the total number of elements in the data.
        - total_blocks: An unsigned integer representing the total number of blocks.
        - threads_per_block: An unsigned long integer representing the total number of threads per block.
        - total_threads: An unsigned long integer representing the total number of threads.
    It modifies the array offsets.
    It follows a recursive approach to distribute the workload evenly among the threads and blocks.
*/
void get_thread_offsets(unsigned long *offsets, const unsigned long long *block_starting_idx, const unsigned long long N, const unsigned total_blocks, const unsigned long threads_per_block, const unsigned long total_threads);