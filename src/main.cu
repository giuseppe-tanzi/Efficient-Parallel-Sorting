#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "../lib/radixSort.cuh"
#include "../lib/mergeSort.cuh"

#define MAXTHREADSPERBLOCK 512
#define MAXBLOCKS 65535

// Useful to check errors in the cuda kernels
#define cudaHandleError(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUerror: %s\nCode: %d\nFile: %s\nLine: %d\n", cudaGetErrorString(code), code, file, line);
        if (abort) exit(code);
    }
}

__global__ void radix_sort_kernel(long int *data, unsigned long n, unsigned offset, const unsigned long n_threads)
{
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long start = tid * offset;
    unsigned old_offset;
    unsigned thread;

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
            start = 0;
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
    }

    if ((n - old_offset) > 0) // MORE THREAD THAN NEEDED
    {
        radix_sort(&data[start], offset);
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
    unsigned temp_n_threads = n_threads;
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
        // radix_sort_kernel<<<gridSize, blockSize>>>(dev_a, N, partition_size, num_total_threads); // GLOBAL MEMORY; TODO: here I could use shared memory with size equal to partition_size
        cudaDeviceSynchronize();
        sort_kernel<<<gridSize, blockSize>>>(dev_a, N, partition_size, num_total_threads); // GLOBAL MEMORY TODO: WRONG!
        cudaDeviceSynchronize();
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