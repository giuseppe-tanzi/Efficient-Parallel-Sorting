#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "../lib/radixSort.cuh"
#include "../lib/mergeSort.cuh"

#define MAXTHREADSPERBLOCK 512
#define MAXBLOCKS 65535

// Useful to check errors in the cuda kernels
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

__global__ void sort_kernel(long int *data, unsigned long n, unsigned offset, const unsigned long n_threads)
{
    // extern __shared__ long int sdata[];
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned thread = 0;
    unsigned long start = tid * offset;
    unsigned long end = start + offset - 1;
    unsigned level_merge = 0;
    unsigned levels_merge = 0;
    unsigned left, mid, right, offset_merge;
    unsigned old_offset;
    unsigned long temp_n_threads = n_threads;
    unsigned threads_to_merge = 0;

    // Compute new start, end and offset for the thread, computing the offset of precedent threads
    if (tid != 0)
    {
        // Compute old offset in a recursive way, in order to compute the start for the thread
        if (tid - 1 == 0)
        {
            start = tid * offset;
        }
        else
        {
            old_offset = offset;
            for (thread = 1; thread < tid; thread++)
            {
                if ((n - old_offset) > 0) // MORE THREAD THAN NEEDED
                {
                    old_offset += (n - old_offset + (n_threads - thread) - 1) / (n_threads - thread);
                }
                else
                {
                    break;
                }
            }
            start = old_offset;
        }
        offset = (n - start + (n_threads - tid) - 1) / (n_threads - tid);
        end = start + offset - 1;
    }

    if ((n - old_offset) > 0) // MORE THREAD THAN NEEDED
    {

        // Log(num_threads)/Log(2) == Log_2(num_threads)
        // Compute number of merge needed in the merge sort
        while (temp_n_threads > 1)
        {
            temp_n_threads /= 2;
            levels_merge++;
        }

        // printf("Sono il thread n.ro %d con last n.ro %lu\n", tid, end);

        // // Load data into shared memory
        // for (long i = start; i < end + 1; i++)
        // {
        //     sdata[i] = data[i];
        // }

        radix_sort(&data[start], offset);
        __syncthreads();

        // Merge the sorted array
        for (level_merge = 1; level_merge <= levels_merge; level_merge++)
        {
            if (level_merge == 1)
            {
                mid = end;
            }

            power(2, level_merge, &threads_to_merge);

            if ((tid % threads_to_merge) == 0)
            {
                left = start;
                offset_merge = offset;

                for (thread = tid + 1; thread < tid + threads_to_merge; thread++)
                {
                    offset_merge += (n - start - offset_merge + (n_threads - thread) - 1) / (n_threads - thread);
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

        // // Write sorted data back to global memory
        // for (long i = start; i < start + offset && i < n; i++)
        // {
        //     data[i] = sdata[i];
        // }
    }
}

unsigned long get_list_to_merge(unsigned long n, unsigned partition, unsigned num_threads)
{
    unsigned i;
    unsigned long offset = partition;
    unsigned long list_to_merge = 1;

    for (i = 1; i < num_threads; i++)
    {
        if ((n - offset) > 0) // MORE THREAD THAN NEEDED
        {
            offset += (n - offset + (num_threads - i) - 1) / (num_threads - i);
            list_to_merge++;
        }
        else
        {
            break;
        }
    }
    return list_to_merge;
}

void get_start_and_size(unsigned long num_block, unsigned long n, unsigned partition, unsigned total_threads, unsigned long *values)
{
    int start = 0;
    int size = 1;
    unsigned long thread;
    unsigned long precedent_threads = num_block * MAXTHREADSPERBLOCK;
    // unsigned long total_threads = (num_block + 1) * MAXTHREADSPERBLOCK;
    unsigned start_v = 0;
    unsigned size_v = 0;

    if (precedent_threads == 0)
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
    for (thread = precedent_threads; thread < (num_block + 1) * MAXTHREADSPERBLOCK; thread++) //TODO: DEBUGGATO E' GIUSTO!
    {
        if ((n - size_v) > 0) // MORE THREAD THAN NEEDED
        {
            size_v += (n - size_v + (total_threads - thread) - 1) / (total_threads - thread);
        }
        else
        {
            break;
        }
    }

    values[size] = size_v - start_v;
}

int main(int argc, char *argv[])
{
    unsigned long N, first, last;
    long int *a, *dev_a;
    unsigned long num_threads_per_block, num_blocks, num_total_threads;
    unsigned partition_size = 50; // TODO: TEMPORARY VALUE
    double tstart, tstop;

    if (argc > 1)
    {
        N = atoi(argv[1]);
    }
    else
    {
        N = 512;
    }

    first = 0;
    last = N - 1;
    const size_t size = N * sizeof(long int);

    a = (long int *)malloc(size);
    cudaHandleError(cudaMalloc((void **)&dev_a, size));

    printf("Sort algorithm on array of %lu elements\n\n", N);

    printf("Sequential implementation:\n");
    init_array(a, N);
    tstart = gettime();
    merge_sort(a, first, last);
    tstop = gettime();
    check_result(a, N);
    printf("Elapsed time in seconds: %f\n\n", (tstop - tstart));

    printf("Parallel implementation:\n");
    init_array(a, N);
    // print_array(a, N);
    cudaHandleError(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));

    if (N <= partition_size) // TODO: RAMO DEBUGGGATO OK
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

        if (num_total_threads <= MAXTHREADSPERBLOCK) // TODO: RAMO DEBUGGATO OK
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
        2. Write a for-loop in which you call each block on a different portion of the array 
        3. cudaDeviceSynchronize();
        3. Call a single block to merge the entire array on the different results of the different blocks
        */

        radix_sort_kernel<<<gridSize, blockSize>>>(dev_a, N, partition_size, num_total_threads); // GLOBAL MEMORY; TODO: here I could use shared memory with size equal to partition_size

        unsigned long n_merge = ceil(get_list_to_merge(N, partition_size, num_total_threads) / 2);
        unsigned long n_blocks_needed = ceil(n_merge / MAXTHREADSPERBLOCK);

        unsigned long **block_dimension;
        block_dimension = (unsigned long **)malloc(n_merge * sizeof(unsigned long *));
        for (int i = 0; i < n_merge; i++)
        {
            block_dimension[i] = (unsigned long *)malloc(2 * sizeof(unsigned long));
        }

        for (int num_block = 0; num_block < n_blocks_needed; num_block++) // TODO: TEST WITH N=25601
        {
            // Compute the size of dev_a and where to start
            get_start_and_size(num_block, N, partition_size, n_blocks_needed * MAXTHREADSPERBLOCK, block_dimension[num_block]);

            // IN QUESTO KERNEL BISOGNA RISALIRE ALLE LISTE ORIGINALI ORDINATE DAL RADIX SORT AFFINCHE' TUTTO FUNZIONI
            // merge_kernel<<<1, blockSize>>>(dev_a[starts[num_block]], sizes[num_block]], partition_size(?), num_threads_per_block); // GLOBAL MEMORY;
        }

        // sort_kernel<<<gridSize, blockSize>>>(dev_a, N, partition_size, num_total_threads); // GLOBAL MEMORY TODO: WRONG!
    }

    tstop = gettime();
    cudaHandleError(cudaPeekAtLastError());
    cudaHandleError(cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost));
    // print_array(a, N);
    check_result(a, N);
    bzero(a, size); /* erase destination buffer, just in case... */
    printf("Elapsed time in seconds: %f\n\n", (tstop - tstart));

    // Free memory on host and device
    free(a);
    cudaHandleError(cudaFree(dev_a));
    return 0;
}