#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>
#include "../lib/radixSort.cuh"
#include "../lib/mergeSort.cuh"

#define MAXTHREADSPERBLOCK 512
#define MAXBLOCKS 65535

/*
    Useful to check errors in the cuda kernels
*/
#define cudaHandleError(ans)                  \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUerror: %s\nCode: %d\nFile: %s\nLine: %d\n", cudaGetErrorString(code), code, file, line);
        if (abort)
            exit(code);
    }
}

/*
    Entire sort kernel:
        1. Radix sort
        2. Merge sort
*/
__global__ void sort_kernel(long int *data, unsigned long n, unsigned offset, const unsigned long n_threads)
{
    // extern __shared__ long int sdata[];
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Variables useful to compute the portion of array for each thread
    unsigned long start = tid * offset;
    unsigned long end = start + offset - 1;
    unsigned old_offset = 0;
    unsigned prec_thread = 0;

    // Variables useful during the merging phase
    unsigned long temp_n_threads = n_threads; // Variable useful to compute the numbers of levels during the merging phase
    unsigned level_merge = 0, levels_merge = 0, offset_merge = 0, threads_to_merge = 0;
    unsigned left = 0, mid = 0, right = 0;

    // Compute new start, end and offset for the thread, computing the offset of precedent threads
    if (tid != 0)
    {
        // Compute old offset in a recursive way, in order to compute the start for the current thread
        if (tid - 1 == 0)
        {
            start = tid * offset;
        }
        else
        {
            old_offset = offset;
            for (prec_thread = 1; prec_thread < tid; prec_thread++)
            {
                /*
                    This if-else is useful if there are more thread than needed:
                        - Ensures that no necessary thread remain in idle
                */
                if ((n - old_offset) > 0)
                {
                    // ceil((n - old_offset/n_threads - prec_thread))
                    old_offset += (n - old_offset + (n_threads - prec_thread) - 1) / (n_threads - prec_thread);
                }
                else
                {
                    break;
                }
            }
            start = old_offset;
        }

        // ceil((n - start) / (n_threads - tid))
        offset = (n - start + (n_threads - tid) - 1) / (n_threads - tid);
        end = start + offset - 1;
    }

    /*
        This if-else is useful if there are more thread than needed:
            - It ensures that no necessary thread remain in idle
    */
    if ((n - old_offset) > 0)
    {

        /*
            Log(num_threads)/Log(2) == Log_2(num_threads)
            Compute number of merge needed in the merge sort
        */
        while (temp_n_threads > 1)
        {
            temp_n_threads /= 2;
            levels_merge++;
        }

        // // Load data into shared memory
        // for (long i = start; i < end + 1; i++)
        // {
        //     sdata[i] = data[i];
        // }

        radix_sort(&data[start], offset);
        __syncthreads();

        // Merge - Phase
        for (level_merge = 1; level_merge <= levels_merge; level_merge++)
        {
            /*
                At first level, mid is equal to the end of the portion sorted by the thread since during the merging phase,
                mid is the final index of the left portion.
            */
            if (level_merge == 1)
            {
                mid = end;
            }

            /*
                threads_to_merge = 2^(level_merge) - Useful to exclude no necessary thread in the successive level
                Threads_to_merge is equal to the number of threads merged from the first level of the merging phase
            */
            power(2, level_merge, &threads_to_merge);

            if ((tid % threads_to_merge) == 0)
            {
                left = start;
                offset_merge = offset;

                /*
                    Useful to compute the size of the resulting list after the current level_merge
                */
                for (prec_thread = tid + 1; prec_thread < tid + threads_to_merge; prec_thread++)
                {
                    // ceil((n - start - offset_merge) / (n_threads - prec_thread))
                    offset_merge += (n - start - offset_merge + (n_threads - prec_thread) - 1) / (n_threads - prec_thread);
                }

                right = left + offset_merge - 1;
                merge(data, left, mid, right);

                /*
                    Merge sort expects to have mid as maximum value of the first list
                    Mid is equal to right to the next level_merge

                */
                mid = right;
            }

            // Needed since the lists to merge to the next level_merge must be ordered
            __syncthreads();
        }

        // // Write sorted data back to global memory
        // for (long i = start; i < start + offset && i < n; i++)
        // {
        //     data[i] = sdata[i];
        // }
    }
}

/*
    Function that returns the number of lists to merge at level 0 of the merging phase
*/
unsigned long get_list_to_merge(unsigned long n, unsigned partition, unsigned num_threads)
{
    unsigned thread = 0;
    unsigned long offset = partition, list_to_merge = 1;

    for (thread = 1; thread < num_threads; thread++)
    {
        if ((n - offset) > 0)
        {
            // ceil((n - offset) / (num_threads - ))
            offset += (n - offset + (num_threads - thread) - 1) / (num_threads - thread);
            list_to_merge++;
        }
        else
        {
            break;
        }
    }

    return list_to_merge;
}

/* COMMENT THIS FUNCTION*/
void get_start_and_size(unsigned long num_block, unsigned long n, unsigned partition, unsigned total_blocks, unsigned total_threads, unsigned long *values, unsigned long *offsets)
{
    unsigned int start = 0;
    unsigned int size = 1;

    unsigned long thread;
    unsigned tid = 0; // Actual thread in the block
    unsigned num_blocks_sort = total_threads / (float)MAXTHREADSPERBLOCK;
    unsigned multiplier = num_blocks_sort / (float)total_blocks;
    unsigned long precedent_threads = multiplier * MAXTHREADSPERBLOCK * num_block;

    // unsigned long total_threads = (num_block + 1) * MAXTHREADSPERBLOCK;
    unsigned start_v = 0;
    unsigned size_v = 0;
    unsigned long offset = 0;

    for (int i = 0; i < MAXTHREADSPERBLOCK; i++)
    {
        offsets[i] = 0;
    }

    if (num_block == 0)
    {
        start_v = 0;
    }
    else
    {
        // Compute start in a recursive way
        for (thread = 0; thread < precedent_threads; thread++)
        {
            if ((n - size_v) > 0) // MORE THREAD THAN NEEDED
            {
                size_v = (n - start_v + (total_threads - thread) - 1) / (total_threads - thread);
                start_v += size_v;
            }
            else
            {
                break;
            }
        }
    }

    values[start] = start_v;

    size_v = start_v;
    // Compute size in a recursive way
    for (thread = precedent_threads; thread < (num_block + 1) * MAXTHREADSPERBLOCK * multiplier; thread++)
    {
        if ((n - size_v) > 0) // MORE THREAD THAN NEEDED
        {
            offset = (n - size_v + (total_threads - thread) - 1) / (total_threads - thread);
            size_v += offset;
            offsets[tid] += offset;
            if (((thread + 1) % multiplier) == 0)
            {
                tid++;
            }
        }
        else
        {
            break;
        }
    }

    values[size] = size_v - start_v;
}

__global__ void merge_kernel(long int *data, unsigned long n, unsigned long *offset, const unsigned long n_threads)
{
    // extern __shared__ long int sdata[];
    const unsigned long tid = threadIdx.x;
    unsigned long start = 0;
    unsigned long end = 0;

    unsigned left, mid, right, offset_merge;
    unsigned level_merge = 0, levels_merge = 0;
    unsigned old_offset;
    unsigned temp_n_threads = n_threads;
    unsigned num_thread_to_merge = 0, threads_to_merge = 0;
    unsigned list_to_merge = 1; // List to merge at level 0

    unsigned long i, j;

    printf("OFFSET: %lu\n", offset[0]);

    // Compute new start, end and offset for the thread, computing the offset of precedent threads
    for (i = 0; i < tid; i++)
    {
        // printf("TID: %d - I=%d - Pippo\n", tid, i);
        start += offset[i];
        printf("TID: %d - N_OFFSET: %d", tid, i);
    }

    printf("TID: %d - START: %d", tid, start);
    end = start + offset[tid] - 1;

    // Log(num_threads)/Log(2) == Log_2(num_threads)
    // Compute number of merge needed in the merge sort
    while (temp_n_threads > 1)
    {
        temp_n_threads /= 2;
        levels_merge++;
    }

    printf("Sono il thread n.ro %lu con last n.ro %lu\n", start, end);

    // // Load data into shared memory
    // for (i = start; i < end + 1; i++)
    // {
    //     sdata[i] = data[i];
    // }

    // Merge the sorted array
    for (level_merge = 0; level_merge < levels_merge; level_merge++)
    {
        if (level_merge == 0)
        {
            mid = end;
        }

        power(2, level_merge, &threads_to_merge);

        if ((tid % threads_to_merge) == 0)
        {
            left = start;
            offset_merge = offset[tid];

            for (num_thread_to_merge = 1; num_thread_to_merge <= threads_to_merge; num_thread_to_merge++)
            {
                offset_merge += offset[tid + num_thread_to_merge];
            }

            right = left + offset_merge - 1;
            merge(data, left, mid, right);
            // printf("TID: %lu - STEP: %d\n", tid, level_merge);
            // printf("LEFT: TID: %lu-%lu\n", tid, left);
            // printf("MID: TID: %lu-%lu\n", tid, mid);
            // printf("RIGHT: TID: %lu-%lu\n", tid, right);
            // printf("OFFSET: TID: %lu-%lu\n", tid, offset_merge);
            // for (long k = start; k < left + offset_merge; k++)
            // {
            //     printf("%lu:%li\n", k, sdata[k]);
            // }

            // Fix since the two merged list are of two different dimension, because the offset is balanced between threads.
            // Merge sort expects to have mid as maximum value of the first list
            mid = right;
        }
        __syncthreads();
    }
}

int main(int argc, char *argv[])
{
    unsigned long N = 512;
    unsigned long first, last;
    long int *a, *dev_a;
    unsigned long num_threads_per_block, num_blocks, num_total_threads;
    unsigned partition_size = 50; // TODO: TEMPORARY VALUE
    double tstart, tstop;

    if (argc > 1)
    {
        N = atoi(argv[1]);
    }

    first = 0;
    last = N - 1;
    const size_t size = N * sizeof(long int);

    a = (long int *)malloc(size);
    cudaHandleError(cudaMalloc((void **)&dev_a, size));

    // Sequential sorting
    printf("Sort algorithm on array of %lu elements\n\n", N);
    printf("Sequential implementation:\n");
    init_array(a, N);
    tstart = gettime();
    merge_sort(a, first, last);
    tstop = gettime();
    check_result(a, N);
    bzero(a, size); // Erase destination buffer
    printf("Elapsed time in seconds: %f\n\n", (tstop - tstart));

    // Parallel sorting
    printf("Parallel implementation:\n");
    init_array(a, N);
    cudaHandleError(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));

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
    if (N <= partition_size)
    {
        num_blocks = 1; // TODO:Depends if the partition size is greater than MAXTHREADPERBLOCK
        for (unsigned long i = N; i >= 2; i--)
        {
            if (IsPowerOfTwo(i))
            {
                num_total_threads = i;
                partition_size = ceil(N / float(num_total_threads));
                num_threads_per_block = num_total_threads;
                break;
            }
        }
    }
    else
    {
        num_total_threads = ceil(N / float(partition_size));

        if (num_total_threads <= MAXTHREADSPERBLOCK)
        {
            num_blocks = 1;
            num_threads_per_block = num_total_threads;

            for (unsigned long i = num_total_threads; i >= 2; i--)
            {
                if (IsPowerOfTwo(i))
                {
                    num_total_threads = i;
                    partition_size = ceil(N / float(num_total_threads));
                    num_threads_per_block = num_total_threads;
                    break;
                }
            }
        }
        else
        {
            num_threads_per_block = MAXTHREADSPERBLOCK;
            num_blocks = ceil(num_total_threads / (float)num_threads_per_block);

            if (num_blocks > MAXBLOCKS)
            {
                num_blocks = MAXBLOCKS;
            }

            num_total_threads = (unsigned long)(num_blocks * num_threads_per_block);

            for (unsigned long i = num_total_threads; i >= 2; i--)
            {
                num_blocks = ceil(i / (float)MAXTHREADSPERBLOCK);
                num_total_threads = (unsigned long)(num_blocks * num_threads_per_block);

                if (IsPowerOfTwo(num_total_threads))
                {
                    partition_size = ceil(N / (float)num_total_threads);
                    break;
                }
            }
        }
    }

    dim3 blockSize(num_threads_per_block);
    dim3 gridSize(num_blocks);

    printf("NUM_THREADS: %lu\n", num_total_threads);
    printf("NUM BLOCKS: %lu\n", num_blocks);
    printf("NUM THREAD PER BLOCK: %lu\n", num_threads_per_block);

    tstart = gettime();
    // sort_kernel<<<gridSize, blockSize, size>>>(dev_a, N, partition_size, num_total_threads); //problem with size shared memory

    /*
        Two different branch to compute the parallel sorting based on the number of blocks
            - First branch: compute the radix sort phase and the merging sort phase in the same kernel
            - Second branch: compute the two phase in two distinct moments
                - The radix sort is computed on the entire array with the all necessary blocks
                - The sorting phase is computed using a different number of blocks, since the number of necessary threads is smaller
                    - By doing so all the threads in each block performs a merge during the first level of the merging phase
                    - Then, the sorting is called on only one block in order to sort all the portion of array sorted by each block
    */
    if (num_blocks == 1)
    {
        sort_kernel<<<gridSize, blockSize>>>(dev_a, N, partition_size, num_total_threads); // GLOBAL MEMORY
    }
    else // since I need that the data is ordered before merge //TODO: PROBLEM WITH 25601
    {
        /*
        STEPS:
        0. Call the radix sort on the array - DONE
        1. Compute the numbers of list to merge - DONE
        2. Get a different portion of the array for each block - DONE
        2. Write a for-loop in which you call each block on a different portion of the array
        3. cudaDeviceSynchronize();
        3. Call a single block to merge the entire array on the different results of the different blocks
        */

        radix_sort_kernel<<<gridSize, blockSize>>>(dev_a, N, partition_size, num_total_threads); // GLOBAL MEMORY; TODO: here I could use shared memory with size equal to partition_size
        cudaHandleError(cudaPeekAtLastError());

        // TODO: DECLARATION VARIABLES AT TOP
        unsigned long n_merge = ceil(get_list_to_merge(N, partition_size, num_total_threads) / 2);
        unsigned long n_blocks_needed = ceil(n_merge / MAXTHREADSPERBLOCK);
        unsigned long **block_dimension = (unsigned long **)malloc(n_blocks_needed * sizeof(unsigned long *));
        unsigned long **thread_offset = (unsigned long **)malloc(n_blocks_needed * sizeof(unsigned long *));
        unsigned long **dev_thread_offset;
        cudaHandleError(cudaMalloc((void **)&dev_thread_offset, n_blocks_needed * sizeof(unsigned long *)));
        unsigned long *block_offset = (unsigned long *)malloc(n_blocks_needed * sizeof(unsigned long));
        for (int i = 0; i < n_blocks_needed; i++)
        {
            block_dimension[i] = (unsigned long *)malloc(2 * sizeof(unsigned long));
            thread_offset[i] = (unsigned long *)malloc(MAXTHREADSPERBLOCK * sizeof(unsigned long));

            if (cudaMallocManaged((void **)&dev_thread_offset[i], MAXTHREADSPERBLOCK * sizeof(unsigned long)) != cudaSuccess)
            {
                printf("Error allocating memory on device for thread_offset[%d]\n", i);
                exit(1);
            }
        }

        const unsigned idx_start = 0;
        const unsigned idx_size = 1;

        for (int num_block = 0; num_block < n_blocks_needed; num_block++) // TODO: TEST WITH N=25601
        {
            // Compute the size of dev_a and where to start
            get_start_and_size(num_block, N, partition_size, n_blocks_needed, num_total_threads, block_dimension[num_block], thread_offset[num_block]);

            for (unsigned long i = 0; i < MAXTHREADSPERBLOCK; i++)
            {
                printf("NUM BLOCK: %d - i: %lu - %lu\n", num_block, i, thread_offset[num_block][i]);
                cudaHandleError(cudaMemcpy(dev_thread_offset[i], thread_offset[i], size, cudaMemcpyHostToDevice));
            }

            cudaHandleError(cudaMemcpy(dev_thread_offset, thread_offset, size, cudaMemcpyHostToDevice));

            // IN QUESTO KERNEL BISOGNA RISALIRE ALLE LISTE ORIGINALI ORDINATE DAL RADIX SORT AFFINCHE' TUTTO FUNZIONI - SICURO SI PUò USARE SHARED MEMORY SUGLI OFFSET
            merge_kernel<<<1, blockSize>>>(&dev_a[block_dimension[num_block][idx_start]], block_dimension[num_block][idx_size], dev_thread_offset[num_block], num_threads_per_block); // GLOBAL MEMORY;
            cudaHandleError(cudaPeekAtLastError());

            block_offset[num_block] = block_dimension[num_block][idx_size] - block_dimension[num_block][idx_start];
        }

        cudaHandleError(cudaDeviceSynchronize());
        cudaHandleError(cudaPeekAtLastError());

        merge_kernel<<<1, blockSize>>>(dev_a, N, block_offset, num_threads_per_block); // GLOBAL MEMORY;
    }

    tstop = gettime();
    cudaHandleError(cudaPeekAtLastError());
    cudaHandleError(cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost));
    check_result(a, N);
    bzero(a, size); // Erase destination buffer
    printf("Elapsed time in seconds: %f\n\n", (tstop - tstart));

    // Free memory on host and device
    free(a);
    cudaHandleError(cudaFree(dev_a));

    return 0;
}