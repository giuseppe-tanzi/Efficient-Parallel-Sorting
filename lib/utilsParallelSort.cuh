#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/mergeSort.cuh"
#include "../lib/radixSort.cuh"
#include "../lib/parallelSort.cuh"
#include "../lib/utils.cuh"

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
    Function that returns the number of lists to merge at level 0 of the merging phase
*/
unsigned long get_n_list_to_merge(unsigned long long N, unsigned long long partition, unsigned long num_threads);

/*
    Function that is responsible for computing the starting index and size of data blocks for the parallel computation,
    given a set of parameters:
        - block_dimension: A pointer to an array of unsigned long integers representing the block dimensions. The starting index and size of each block will be stored in this array.
        - offsets: A pointer to an array of unsigned long integers representing the offsets for each thread in each block. The offset for each thread will be accumulated in this array.
        - N: An unsigned long integer representing the total number of elements in the data.
        - partition: An unsigned integer representing the size of the partition.
        - total_blocks: An unsigned integer representing the total number of blocks.
        - total_threads: An unsigned integer representing the total number of threads.
    It operates on unsigned long integers and modifies two arrays: block_dimension and offsets.
    It follows a recursive approach to distribute the workload evenly among the threads and blocks
*/
__host__ __device__ void get_start_and_size(unsigned long long *block_dimension, unsigned long *offsets, unsigned long long N, unsigned long long partition, unsigned total_blocks, unsigned long total_threads);