#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>
#include "../lib/radixSort.cuh"
#include "../lib/mergeSort.cuh"
#include "../lib/parallelSort.cuh"
#include "../lib/utils.cuh"

/*
    Function that returns the number of lists to merge at level 0 of the merging phase
*/
unsigned long get_n_list_to_merge(unsigned long long N, unsigned long long partition, unsigned long num_threads)
{
    unsigned long thread = 0;
    unsigned long long offset = partition, n_list_to_merge = 1;

    for (thread = 1; thread < num_threads; thread++)
    {
        if ((N - offset) > 0)
        {
            // ceil((n - offset) / (num_threads - ))
            offset += (N - offset + (num_threads - thread) - 1) / (num_threads - thread);
            n_list_to_merge++;
        }
        else
        {
            break;
        }
    }

    return n_list_to_merge;
}

/*
    Function that is responsible for computing the starting index and size of data blocks for the parallel computation,
    given a set of parameters:
        - block_dimension: A pointer to an array of unsigned long integers representing the block dimensions. The starting index and size of each block will be stored in this array.
        - offsets: A pointer to an array of unsigned long integers representing the offsets for each thread in each block. The offset for each thread will be accumulated in this array.
        - n: An unsigned long integer representing the total number of elements in the data.
        - partition: An unsigned integer representing the size of the partition.
        - total_blocks: An unsigned integer representing the total number of blocks.
        - total_threads: An unsigned integer representing the total number of threads.
    It operates on unsigned long integers and modifies two arrays: block_dimension and offsets.
    It follows a recursive approach to distribute the workload evenly among the threads and blocks
*/
__host__ __device__ void get_start_and_size(unsigned long long *block_dimension, unsigned long *offsets, unsigned long long N, unsigned long long partition, unsigned total_blocks, unsigned long total_threads)
{
    unsigned long long idx_start = 0;
    unsigned long idx_size = 0;
    unsigned idx_tid = 0; // Actual thread in the block

    unsigned long thread = 0;
    unsigned num_blocks_sort = total_threads / (float)MAXTHREADSPERBLOCK;
    unsigned multiplier = num_blocks_sort / (float)total_blocks;
    unsigned long precedent_threads = multiplier * MAXTHREADSPERBLOCK;
    unsigned current_block = 0;

    // unsigned long total_threads = (num_block + 1) * MAXTHREADSPERBLOCK;
    unsigned long long start_v = 0;
    unsigned long size_v = 0;
    unsigned long long offset = 0;

    // Initialization of the offset for each thread in each block
    for (unsigned i = 0; i < total_blocks * MAXTHREADSPERBLOCK; i++)
    {
        offsets[i] = 0;
    }

    for (current_block = 0; current_block < total_blocks; current_block++)
    {
        precedent_threads = multiplier * MAXTHREADSPERBLOCK * current_block;
        idx_start = current_block * 2;
        idx_size = idx_start + 1;
        idx_tid = current_block * MAXTHREADSPERBLOCK;
        start_v = 0;
        size_v = 0;

        if (current_block == 0)
        {
            start_v = 0;
        }
        else
        {
            // Compute start in a recursive way
            for (thread = 0; thread < precedent_threads; thread++)
            {
                if ((N - size_v) > 0) // MORE THREAD THAN NEEDED
                {
                    size_v = (N - start_v + (total_threads - thread) - 1) / (total_threads - thread);
                    start_v += size_v;
                }
                else
                {
                    break;
                }
            }
        }

        block_dimension[idx_start] = start_v;

        size_v = start_v;
        // Compute size in a recursive way
        for (thread = precedent_threads; thread < (current_block + 1) * MAXTHREADSPERBLOCK * multiplier; thread++)
        {
            if ((N - size_v) > 0) // MORE THREAD THAN NEEDED
            {
                offset = (N - size_v + (total_threads - thread) - 1) / (total_threads - thread);
                size_v += offset;
                offsets[idx_tid] += offset;
                if (((thread + 1) % multiplier) == 0)
                {
                    idx_tid++;
                }
            }
            else
            {
                break;
            }
        }

        block_dimension[idx_size] = size_v - start_v;
    }
}

/*
    The merge_kernel function is a CUDA kernel that performs a merge sort on an array data in parallel.
    It divides the array into smaller ranges and merges them progressively until the entire array is sorted.
*/
__global__ void merge_kernel(unsigned short *data, unsigned long long n, const unsigned long *offset, const unsigned long n_threads, const unsigned long num_threads_precedent_blocks)
{
    // extern __shared__ long int sdata[];
    const unsigned long tid = num_threads_precedent_blocks + threadIdx.x;
    unsigned long long start = 0;
    unsigned long long end = 0;

    unsigned long long left, mid, right, offset_merge;
    unsigned level_merge = 0, levels_merge = 0;
    unsigned long temp_n_threads = n_threads;
    unsigned num_thread_to_merge = 0, threads_to_merge = 0;

    unsigned long i;

    // Compute new start, end and offset for the thread, computing the offset of precedent threads
    for (i = num_threads_precedent_blocks; i < tid; i++)
    {
        start += offset[i];
    }

    end = start + offset[tid] - 1;

    // Log(n_threads)/Log(2) == Log_2(n_threads)
    // Compute number of merge needed in the merge sort
    while (temp_n_threads > 1)
    {
        temp_n_threads /= 2;
        levels_merge++;
    }

    // printf("Sono il thread n.ro %lu con last n.ro %lu\n", start, end);

    // // Load data into shared memory
    // for (i = start; i < end + 1; i++)
    // {
    //     sdata[i] = data[i];
    // }

    // Merge the sorted array
    for (level_merge = 0; level_merge <= levels_merge; level_merge++)
    {

        power(2, level_merge, &threads_to_merge);

        if ((tid % threads_to_merge) == 0)
        {
            left = start;
            offset_merge = offset[tid];

            for (num_thread_to_merge = 1; num_thread_to_merge < threads_to_merge; num_thread_to_merge++)
            {
                offset_merge += offset[tid + num_thread_to_merge];
            }

            right = left + offset_merge - 1;

            if (level_merge == 0)
            {
                mid = left + (right - left) / 2; // TODO: FIX MID MAYBE IT IS WRONG
            }

            merge_dev(data, left, mid, right);
            // if (tid == 512 && level_merge == levels_merge)
            // {
            //     printf("STEP: %d - TID: %d - RIGHT: %lu\n", level_merge, tid, right);
            //     printf("STEP: %d - TID: %d - LEFT: %lu\n", level_merge, tid, left);
            //     printf("STEP: %d - TID: %d - OFFSET_MERGE: %lu\n", level_merge, tid, offset_merge);
            //     printf("STEP: %d - TID: %d - MID: %lu\n", level_merge, tid, mid);
            //     for (long k = left; k < offset_merge; k++)
            //     {
            //         printf("%lu:%li\n", k, data[k]);
            //     }
            // }

            // Fix since the two merged list are of two different dimension, because the offset is balanced between threads.
            // Merge sort expects to have mid as maximum value of the first list
            mid = right;
        }
        __syncthreads();
    }
}

// TODO:Comment this function
__global__ void merge_blocks_lists_kernel(unsigned short *data, unsigned long *thread_offset, unsigned long long N, ParallelSortConfig config, const unsigned n_threads)
{
    // extern __shared__ long int sdata[];
    const unsigned tid = threadIdx.x;
    unsigned long long start = 0;
    unsigned long long end = 0;
    unsigned long long *block_mid;
    unsigned long long *block_offset;
    unsigned long long *block_dimension;
    unsigned totalBlocks = n_threads * 2;

    unsigned long long left, mid, right, offset_merge;
    unsigned level_merge = 0, levels_merge = 0;
    unsigned long temp_n_threads = n_threads;
    unsigned num_thread_to_merge = 0, threads_to_merge = 0;

    unsigned long long idx_block_start = 0;
    unsigned long idx_block_size = 0;
    unsigned long idx_next_block_size = 0;

    unsigned i;

    cudaHandleErrorGPU(cudaMalloc((void **)&block_dimension, totalBlocks * sizeof(unsigned long long)));
    cudaHandleErrorGPU(cudaMalloc((void **)&block_offset, n_threads * sizeof(unsigned long long)));
    cudaHandleErrorGPU(cudaMalloc((void **)&block_mid, n_threads * sizeof(unsigned long long)));

    get_start_and_size(block_dimension, thread_offset, N, config.partitionSize, totalBlocks, config.nTotalThreads);

    for (unsigned num_block = 0; num_block < totalBlocks; num_block++)
    {
        idx_block_start = num_block * 2;
        idx_block_size = idx_block_start + 1;

        if ((num_block % 2) == 0)
        {
            idx_next_block_size = (num_block + 1) * 2 + 1;
            // Add the offset of the successive block
            block_offset[num_block / 2] = block_dimension[idx_block_size] + block_dimension[idx_next_block_size];

            // Compute mid useful during the first level merge
            block_mid[num_block / 2] = 0;
            for (unsigned i = 0; i <= num_block; i++)
            {
                block_mid[num_block / 2] += block_dimension[i * 2 + 1];
            }
        }
    }

    // Compute new start, end and offset for the thread, computing the offset of precedent threads
    for (i = 0; i < tid; i++)
    {
        start += block_offset[i];
    }

    end = start + block_offset[tid] - 1;
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
            merge_dev(data, left, mid, right);

            // Fix since the two merged list are of two different dimension, because the offset is balanced between threads.
            // Merge sort expects to have mid as maximum value of the first list
            mid = right;
        }
        __syncthreads();
    }

    cudaHandleErrorGPU(cudaFree(block_dimension));
    cudaHandleErrorGPU(cudaFree(block_offset));
    cudaHandleErrorGPU(cudaFree(block_mid));
}

void parallel_sort(unsigned short *dev_a,
                   const unsigned long long N,
                   ParallelSortConfig config,
                   const size_t size_blocks,
                   unsigned nBlocksMerge,
                   unsigned long long *block_dimension,
                   unsigned long *thread_offset,
                   unsigned long *dev_thread_offset)
{

    unsigned long long idx_block_start = 0;
    unsigned long idx_block_size = 0;

    /*
        Two different branch to compute the parallel sorting based on the number of blocks
            - First branch: compute the radix sort phase and the merging sort phase in the same kernel
            - Second branch: compute the two phase in two distinct moments
                - The radix sort is computed on the entire array with the all necessary blocks
                - The merging phase is computed using a different number of blocks, since the number of necessary threads is smaller
                    - By doing so all the threads in each block performs a merge during the first level of the merging phase
                    - Then, the sorting is called on only one block in order to sort all the portion of array sorted by each block
    */

    if (config.nBlocks == 1)
    {
        sort_kernel<<<config.gridSize, config.blockSize>>>(dev_a, N, config.partitionSize, config.nTotalThreads); // GLOBAL MEMORY
    }
    else
    {
        /*
        STEPS:
        0. Call the radix sort on the array - DONE
        1. Compute the numbers of list to merge - DONE
        2. Get a different portion of the array for each block - DONE
        3. Write a for-loop in which you call each block on a different portion of the array
        4. cudaDeviceSynchronize();
        5. Call a single block to merge the entire array on the different results of the different blocks
        */

        // The data has to be ordered before merging phase
        radix_sort_kernel<<<config.gridSize, config.blockSize>>>(dev_a, N, config.partitionSize, config.nTotalThreads); // GLOBAL MEMORY; TODO: here I could use shared memory with size equal to partition_size
        cudaHandleError(cudaDeviceSynchronize());
        cudaHandleError(cudaPeekAtLastError());

        // Compute the size of dev_a and where to start
        get_start_and_size(block_dimension, thread_offset, N, config.partitionSize, nBlocksMerge, config.nTotalThreads);
        cudaHandleError(cudaMemcpy(dev_thread_offset, thread_offset, size_blocks, cudaMemcpyHostToDevice));
        // for (unsigned long i = 0; i < n_blocks_merge * MAXTHREADSPERBLOCK; i++)
        // {
        //     printf("%lu:%li\n", i, thread_offset[i]);
        // }
        printf("N BLOCKs MERGE: %d\n", nBlocksMerge);

        for (unsigned num_block = 0; num_block < nBlocksMerge; num_block++)
        {
            idx_block_start = num_block * 2;
            idx_block_size = idx_block_start + 1;
            // printf("NUM BLOCK MERGE: %d\n", num_block);
            // printf("START: %lu\n", block_dimension[idx_block_start]);
            // printf("SIZE %lu\n", block_dimension[idx_block_size]);

            // cudaHandleError(cudaMemcpy(a, dev_a, size_array, cudaMemcpyDeviceToHost));
            // print_array(a + block_dimension[idx_block_start], block_dimension[idx_block_size]);

            // TODO: SICURO SI PUò USARE SHARED MEMORY SUGLI OFFSET
            merge_kernel<<<1, config.blockSize>>>(dev_a + block_dimension[idx_block_start], block_dimension[idx_block_size], dev_thread_offset, config.nThreadsPerBlock, num_block * MAXTHREADSPERBLOCK); // GLOBAL MEMORY;
        }

        if (nBlocksMerge > 1)
         {
            // TODO: PROBLEMS IF THE BLOCKS TO MERGE ARE MORE THAN THE MAXIMUM NUMBER OF THREADS IN A SINGLE BLOCK
            merge_blocks_lists_kernel<<<1, nBlocksMerge / 2>>>(dev_a, dev_thread_offset, N, config, nBlocksMerge / 2); // GLOBAL MEMORY;
        }
    }
}

int main(int argc, char *argv[])
{
    unsigned long long N = 512;
    unsigned short *a, *dev_a;

    // Variables useful for parallel
    ParallelSortConfig sortConfig;
    unsigned long nMerge = 0;
    unsigned nBlocksMerge = 0;
    unsigned long long *block_dimension;
    unsigned long *thread_offset;
    unsigned long *dev_thread_offset;

    double tstart = 0, tstop = 0;

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
    tstart = gettime();
    merge_sort(a, 0, N - 1);
    tstop = gettime();
    check_result(a, N);
    bzero(a, size_array); // Erase destination buffer
    printf("Elapsed time in seconds: %f\n\n", (tstop - tstart));

    // Parallel sorting
    printf("Parallel implementation:\n");
    init_array(a, N);
    cudaHandleError(cudaMemcpy(dev_a, a, size_array, cudaMemcpyHostToDevice));

    tstart = gettime();

    // Determine block and thread configurations
    sortConfig = determine_config(N);

    sortConfig.blockSize = dim3(sortConfig.nThreadsPerBlock);
    sortConfig.gridSize = dim3(sortConfig.nBlocks);

    nMerge = ceil(get_n_list_to_merge(N, sortConfig.partitionSize, sortConfig.nTotalThreads) / (float)2);
    nBlocksMerge = ceil(nMerge / (float)MAXTHREADSPERBLOCK);
    const size_t size_blocks = nBlocksMerge * MAXTHREADSPERBLOCK * sizeof(unsigned long);

    // It contains the start id and the size to handle in the array, of each block
    block_dimension = (unsigned long long *)malloc(nBlocksMerge * 2 * sizeof(unsigned long long));

    thread_offset = (unsigned long *)malloc(size_blocks);
    cudaHandleError(cudaMalloc((void **)&dev_thread_offset, size_blocks));

    // printf("NUM_THREADS: %lu\n", sortConfig.nTotalThreads);
    // printf("NUM BLOCKS: %lu\n", sortConfig.nBlocks);
    // printf("NUM THREAD PER BLOCK: %lu\n", sortConfig.nThreadsPerBlock);
    // printf("PARTITION SIZE: %llu\n", sortConfig.partitionSize);

    parallel_sort(dev_a, N, sortConfig, size_blocks, nBlocksMerge, block_dimension, thread_offset, dev_thread_offset);
    tstop = gettime();

    cudaHandleError(cudaPeekAtLastError());
    cudaHandleError(cudaMemcpy(a, dev_a, size_array, cudaMemcpyDeviceToHost));
    check_result(a, N);
    bzero(a, size_array); // Erase destination buffer
    printf("Elapsed time in seconds: %f\n\n", (tstop - tstart));

    // Cleanup
    free(a);
    free(block_dimension);
    free(thread_offset);
    cudaHandleError(cudaFree(dev_thread_offset));
    cudaHandleError(cudaFree(dev_a));

    return 0;
}