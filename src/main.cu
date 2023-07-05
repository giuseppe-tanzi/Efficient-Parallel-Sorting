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

// TODO:Comment this function
__global__ void merge_blocks_lists_kernel(unsigned short *data, unsigned long long N, ParallelSortConfig config, const unsigned n_threads)
{
    // extern __shared__ long int sdata[];
    const unsigned tid = threadIdx.x;
    unsigned long long start = 0;

    unsigned long long *block_starting_idx;
    unsigned long long *block_size;
    unsigned long long *block_offset;
    unsigned long long *block_mid;
    unsigned totalBlocks = n_threads * 2;

    unsigned long long left, mid, right, offset_merge;
    unsigned level_merge = 0, levels_merge = 0;
    unsigned long temp_n_threads = n_threads;
    unsigned num_thread_to_merge = 0, threads_to_merge = 0;

    unsigned i;

    cudaHandleErrorGPU(cudaMalloc((void **)&block_starting_idx, totalBlocks * sizeof(unsigned long long)));
    cudaHandleErrorGPU(cudaMalloc((void **)&block_size, totalBlocks * sizeof(unsigned long long)));
    cudaHandleErrorGPU(cudaMalloc((void **)&block_offset, n_threads * sizeof(unsigned long long)));
    cudaHandleErrorGPU(cudaMalloc((void **)&block_mid, n_threads * sizeof(unsigned long long)));

    /*
        - Compute the start index on the data array for each block
        - Compute the size of the data array to handle for each block
    */
    get_start_index_block(block_starting_idx, N, totalBlocks, config.threads_per_block, config.total_threads);
    get_size_block(block_size, block_starting_idx, N, totalBlocks, config.threads_per_block, config.total_threads);

    for (unsigned block = 0; block < totalBlocks; block++)
    {

        if ((block % 2) == 0)
        {
            // Add the offset of the successive block
            block_offset[block / 2] = block_size[block] + block_size[block + 1];

            // Compute mid useful during the first level merge
            block_mid[block / 2] = 0;
            for (i = 0; i <= block; i++)
            {
                block_mid[block / 2] += block_size[i];
            }
        }
    }

    // Compute new start, end and offset for the thread, computing the offset of precedent threads
    for (i = 0; i < tid; i++)
    {
        start += block_offset[i];
    }

    mid = block_mid[tid] - 1;

    // printf("Sono il thread n.ro %lu con last n.ro %lu\n", start, end);
    // printf("TOTALS LEVEL MERGE: %d\n", levels_merge);

    // // Load data into shared memory
    // for (i = start; i < end + 1; i++)
    // {
    //     sdata[i] = data[i];
    // }

    // Log(n_threads)/Log(2) == Log_2(n_threads)
    // Compute number of merge needed in the merge sort
    while (temp_n_threads > 1)
    {
        temp_n_threads /= 2;
        levels_merge++;
    }

    // Merge the sorted array
    for (level_merge = 0; level_merge <= levels_merge; level_merge++)
    {

        power(2, level_merge, &threads_to_merge);

        if ((tid % threads_to_merge) == 0)
        {
            left = start;
            offset_merge = block_offset[tid];

            for (num_thread_to_merge = 1; num_thread_to_merge < threads_to_merge; num_thread_to_merge++)
            {
                offset_merge += block_offset[tid + num_thread_to_merge];
            }

            right = left + offset_merge - 1;

            // printf("STEP: %d - TID: %d - RIGHT: %llu\n", level_merge, tid, right);
            // printf("STEP: %d - TID: %d - MID: %llu\n", level_merge, tid, mid);
            // printf("STEP: %d - TID: %d - LEFT: %llu\n", level_merge, tid, left);
            // printf("STEP: %d - TID: %d - OFFSET_MERGE: %llu\n", level_merge, tid, offset_merge);
            // printf("\n");
            // if (tid != 1)
            // {
            //     for (long long k = start; k < left + offset_merge; k++)
            //     {
            //         printf("%llu:%hu\n", k, data[k]);
            //     }
            // }

            // printf("AFTER\n");
            merge_dev(data, left, mid, right);

            // Fix since the two merged list are of two different dimension, because the offset is balanced between threads.
            // Merge sort expects to have mid as maximum value of the first list
            mid = right;
        }
        __syncthreads();
    }

    cudaHandleErrorGPU(cudaFree(block_starting_idx));
    cudaHandleErrorGPU(cudaFree(block_size));
    cudaHandleErrorGPU(cudaFree(block_offset));
    cudaHandleErrorGPU(cudaFree(block_mid));
}

/*
    Function that performs the parallel sorting
    Two different branch to compute the parallel sorting based on the number of blocks:
        - First branch: compute the radix sort phase and the merging sort phase in the same kernel
        - Second branch: compute the sorting in two different phase
            - The radix sort is computed on the entire array with the all necessary blocks
            - The merging phase is computed using a different number of blocks, since the number of necessary threads is smaller
                - By doing so all the threads in each block performs a merge during the first level of the merging phase
                - Then, the sorting is called on only one block in order to sort all the portion of array sorted by each block
*/
void parallel_sort(unsigned short *dev_a,
                   const unsigned long long N,
                   ParallelSortConfig config,
                   const size_t size_blocks,
                   unsigned blocks_involved_in_merging,
                   unsigned long long *block_starting_idx,
                   unsigned long long *block_size,
                   unsigned long *thread_offset,
                   unsigned long *dev_thread_offset)
{

    if (config.total_blocks == 1)
    {
        sort_kernel<<<config.gridSize, config.blockSize>>>(dev_a, N, config.partition_size, config.total_threads); // GLOBAL MEMORY
    }
    else
    {
        // The data has to be sorted before merging phase
        radix_sort_kernel<<<config.gridSize, config.blockSize>>>(dev_a, N, config.partition_size, config.total_threads); // GLOBAL MEMORY; TODO: here I could use shared memory with size equal to partition_size
        cudaHandleError(cudaDeviceSynchronize());
        cudaHandleError(cudaPeekAtLastError());

        /*
            - Compute the start index on the data array for each block
            - Compute the offset on the data array to handle for each thread of each needed block
            - Compute the size of the data array to handle for each block
        */ 
        get_start_index_block(block_starting_idx, N, blocks_involved_in_merging, config.threads_per_block, config.total_threads);
        get_thread_offsets(thread_offset, block_starting_idx, N, blocks_involved_in_merging, config.threads_per_block, config.total_threads);
        get_size_block(block_size, block_starting_idx, N, blocks_involved_in_merging, config.threads_per_block, config.total_threads);

        cudaHandleError(cudaMemcpy(dev_thread_offset, thread_offset, size_blocks, cudaMemcpyHostToDevice));

        /*
            It calls the merge kernel on each block
            Each block has a defined portion of the array to handle and a precise number of lists to merge
            The array will have blocks_involved_in_merging lists to merge at the end of the for-loop
        */
        for (unsigned block = 0; block < blocks_involved_in_merging; block++)
        {

            // TODO: SICURO SI PUò USARE SHARED MEMORY SUGLI OFFSET
            merge_kernel<<<1, config.blockSize>>>(dev_a + block_starting_idx[block], dev_thread_offset, config.threads_per_block, block * config.threads_per_block); // GLOBAL MEMORY;
        }
        /*
            It performs the last merging phase with only one block since the number of lists to sort are surely
            less than the maximum thread number for each block
        */
        if (blocks_involved_in_merging > 1)
        {
            merge_blocks_lists_kernel<<<1, blocks_involved_in_merging / 2>>>(dev_a, N, config, blocks_involved_in_merging / 2); // GLOBAL MEMORY;
        }
    }
}

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
        First I get how many sorted list I have at level 0
        Then I divide by 2 to have the number of merge that I'm going to have at level 0
    */
    lists_to_merge = ceil(get_n_list_to_merge(N, sort_config.partition_size, sort_config.total_threads) / (float)2);

    /*
        The number of blocks needed during the merging phase
    */
    blocks_involved_in_merging = ceil(lists_to_merge / (float)sort_config.threads_per_block);
    const size_t size_blocks = blocks_involved_in_merging * sort_config.threads_per_block * sizeof(unsigned long);

    /*
        It contains the start index in the array for each block
    */
    block_starting_idx = (unsigned long long *)malloc(blocks_involved_in_merging * sizeof(unsigned long long));

    /*
        It contains the size to handle in the data array for each block
    */
    block_size = (unsigned long long *)malloc(blocks_involved_in_merging * sizeof(unsigned long long));

    thread_offset = (unsigned long *)malloc(size_blocks);
    cudaHandleError(cudaMalloc((void **)&dev_thread_offset, size_blocks));

    printf("NUM_THREADS: %lu\n", sort_config.total_threads);
    printf("NUM BLOCKS: %lu\n", sort_config.total_blocks);
    printf("NUM THREAD PER BLOCK: %lu\n", sort_config.threads_per_block);
    printf("NUM BLOCKS MERGE: %d\n", blocks_involved_in_merging);
    printf("PARTITION SIZE: %llu\n", sort_config.partition_size);

    parallel_sort(dev_a, N, sort_config, size_blocks, blocks_involved_in_merging, block_starting_idx, block_size, thread_offset, dev_thread_offset);

    t_stop = get_time();

    cudaHandleError(cudaPeekAtLastError());
    cudaHandleError(cudaMemcpy(a, dev_a, size_array, cudaMemcpyDeviceToHost));

    check_result(a, N);
    bzero(a, size_array); // Erase destination buffer

    printf("Elapsed time in seconds: %f\n\n", (t_stop - t_start));

    // Cleanup
    free(a);
    free(block_starting_idx);
    free(thread_offset);
    cudaHandleError(cudaFree(dev_thread_offset));
    cudaHandleError(cudaFree(dev_a));

    return 0;
}