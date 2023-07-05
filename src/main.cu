#include <stdio.h>
#include <stdlib.h>
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

    // Sequential sorting
    printf("Sort algorithm on array of %llu elements\n\n", N);
    printf("Sequential implementation:\n");
    init_array(a, N);
    t_start = get_time();
    merge_sort(a, 0, N - 1);
    t_stop = get_time();
    check_result(a, N);
    bzero(a, size_array); // Erase destination buffer
    printf("Elapsed time in seconds: %f\n\n", (t_stop - t_start));

    // Parallel sorting
    printf("Parallel implementation:\n");
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

    check_result(a, N);
    bzero(a, size_array); // Erase destination buffer

    printf("NUM_THREADS: %lu\n", sort_config.total_threads);
    printf("NUM BLOCKS: %lu\n", sort_config.total_blocks);
    printf("NUM THREAD PER BLOCK: %lu\n", sort_config.threads_per_block);
    printf("NUM BLOCKS MERGE: %d\n", blocks_involved_in_merging);
    printf("PARTITION SIZE: %llu\n", sort_config.partition_size);
    printf("Elapsed time in seconds: %f\n\n", (t_stop - t_start));

    // Cleanup
    free(a);
    free(block_starting_idx);
    free(thread_offset);
    cudaHandleError(cudaFree(dev_thread_offset));
    cudaHandleError(cudaFree(dev_a));

    return 0;
}