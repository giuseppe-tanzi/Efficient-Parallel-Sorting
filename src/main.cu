#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>
#include "../lib/radixSort.cuh"
#include "../lib/mergeSort.cuh"
#include "../lib/parallelSort.cuh"
#include "../lib/utils.cuh"
#include "../lib/utilsParallelSort.cuh"

int main(int argc, char *argv[])
{
    unsigned short n_algorithms = 3;
    char algorithms[][100] = {"Sequential Radix Sort", 
                            "Sequential Merge Sort", 
                            "Parallel Radix Sort + Merge Sort \\w global"};
                            // "Parallel Radix Sort + Merge Sort '\w shared\t(GPU)"};
    char machine[][100] = {"CPU", "CPU", "GPU"};
    unsigned long threads[n_algorithms];
    bool correctness[n_algorithms];
    double elapsed_time[n_algorithms];

    unsigned long long N = 512;
    unsigned short *a, *dev_a;

    // Variables useful for parallel sorting
    ParallelSortConfig sort_config;
    unsigned long lists_to_merge = 0;
    unsigned blocks_involved_in_merging = 0;
    unsigned long long *block_starting_idx;
    unsigned long long *block_size;
    unsigned long *thread_offset;
    unsigned long *dev_thread_offset;

    double t_start = 0, t_stop = 0;

    if (argc > 1)
    {
        N = atoi(argv[1]);
    }

    const size_t size_array = N * sizeof(unsigned short);
    a = (unsigned short *)malloc(size_array);
    cudaHandleError(cudaMalloc((void **)&dev_a, size_array));

    // Sequential sorting with Radix Sort
    printf("Sort algorithm on array of %llu elements\n\n", N);
    init_array(a, N);
    t_start = get_time();
    radix_sort(a, N);
    t_stop = get_time();
    threads[0] = 1;
    correctness[0] = is_sorted(a, N);
    elapsed_time[0] = t_stop - t_start;
    bzero(a, size_array); // Erase destination buffer

    // Sequential sorting with Merge Sort
    init_array(a, N);
    t_start = get_time();
    merge_sort(a, 0, N - 1);
    t_stop= get_time();
    threads[1] = 1;
    correctness[1] = is_sorted(a, N);
    elapsed_time[1] = t_stop - t_start;
    bzero(a, size_array); // Erase destination buffer

    // Parallel sorting
    init_array(a, N);
    cudaHandleError(cudaMemcpy(dev_a, a, size_array, cudaMemcpyHostToDevice));

    t_start = get_time();

    // Determine block and thread configurations
    sort_config = determine_config(N);

    sort_config.blockSize = dim3(sort_config.threads_per_block);
    sort_config.gridSize = dim3(sort_config.total_blocks);

    /*
        - First I get how many sorted list I have at level 0
        - Then I divide by 2 to have the number of merge that I'm going to have at level 0
    */
    lists_to_merge = ceil(get_n_list_to_merge(N, sort_config.partition_size, sort_config.total_threads) / (float)2);

    /*
        - The number of blocks needed during the merging phase
    */
    blocks_involved_in_merging = ceil(lists_to_merge / (float)sort_config.threads_per_block);
    const size_t size_blocks = blocks_involved_in_merging * sort_config.threads_per_block * sizeof(unsigned long);

    /*
        - It contains the start index in the array for each block
    */
    block_starting_idx = (unsigned long long *)malloc(blocks_involved_in_merging * sizeof(unsigned long long));

    /*
        - It contains the size to handle in the data array for each block
    */
    block_size = (unsigned long long *)malloc(blocks_involved_in_merging * sizeof(unsigned long long));

    thread_offset = (unsigned long *)malloc(size_blocks);
    cudaHandleError(cudaMalloc((void **)&dev_thread_offset, size_blocks));

    parallel_sort(dev_a, N, sort_config, size_blocks, blocks_involved_in_merging, block_starting_idx, block_size, thread_offset, dev_thread_offset);

    t_stop = get_time();

    cudaHandleError(cudaPeekAtLastError());
    cudaHandleError(cudaMemcpy(a, dev_a, size_array, cudaMemcpyDeviceToHost));

    threads[2] = sort_config.total_threads;
    correctness[2] = is_sorted(a, N);
    elapsed_time[2] = t_stop - t_start;
    bzero(a, size_array); // Erase destination buffer

    // printf("NUM_THREADS: %lu\n", sort_config.total_threads);
    // printf("NUM BLOCKS: %lu\n", sort_config.total_blocks);
    // printf("NUM THREAD PER BLOCK: %lu\n", sort_config.threads_per_block);
    // printf("NUM BLOCKS MERGE: %d\n", blocks_involved_in_merging);
    // printf("PARTITION SIZE: %llu\n", sort_config.partition_size);

    // Print the table
    print_table(n_algorithms, algorithms, machine, threads, correctness, elapsed_time);

    // Cleanup
    free(a);
    free(block_starting_idx);
    free(thread_offset);
    cudaHandleError(cudaFree(dev_thread_offset));
    cudaHandleError(cudaFree(dev_a));

    return 0;
}