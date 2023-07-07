#include "../lib/mergeSort.cuh"

void merge(unsigned short *data, const unsigned long long left, const unsigned long long mid, const unsigned long long right)
{
    unsigned long long i, j, k;
    unsigned long long dim_left = mid - left + 1;
    unsigned long long dim_right = right - mid;

    /* Create temp arrays */
    unsigned short *temp_left = (unsigned short *)malloc(dim_left * sizeof(unsigned short));
    unsigned short *temp_right = (unsigned short *)malloc(dim_right * sizeof(unsigned short));

    /* Copy data to temp arrays temp_left[] and temp_right[] */
    for (i = 0; i < dim_left; i++)
    {
        temp_left[i] = data[left + i];
    }

    for (j = 0; j < dim_right; j++)
    {
        temp_right[j] = data[mid + 1 + j];
    }

    /* Merge the temp arrays back into data[l..r]*/
    i = 0;    // Initial index of first subarray
    j = 0;    // Initial index of second subarray
    k = left; // Initial index of merged subarray
    while (i < dim_left && j < dim_right)
    {
        if (temp_left[i] <= temp_right[j])
        {
            data[k] = temp_left[i];
            i++;
        }
        else
        {
            data[k] = temp_right[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of temp_left[], if there are any */
    while (i < dim_left)
    {
        data[k] = temp_left[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of temp_right[], if there are any */
    while (j < dim_right)
    {
        data[k] = temp_right[j];
        j++;
        k++;
    }

    /* Free memory */
    free(temp_left);
    free(temp_right);
}

__device__ void merge_gpu(unsigned short *data, const unsigned long long left, const unsigned long long mid, const unsigned long long right)
{
    unsigned long long i, j, k;
    unsigned long long dim_left = mid - left + 1;
    unsigned long long dim_right = right - mid;
    unsigned short *temp_left;
    unsigned short *temp_right;

    /* Create temp arrays */
    cudaHandleErrorGPU(cudaMalloc((void**)&temp_left, dim_left * sizeof(unsigned short))); 
    cudaHandleErrorGPU(cudaMalloc((void**)&temp_right, dim_right * sizeof(unsigned short)));
    
    /* Copy data to temp arrays temp_left[] and temp_right[] */
    for (i = 0; i < dim_left; i++)
    {
        temp_left[i] = data[left + i];
    }

    for (j = 0; j < dim_right; j++)
    {
        temp_right[j] = data[mid + 1 + j];
    }

    /* Merge the temp arrays back into data[l..r]*/
    i = 0;    // Initial index of first subarray
    j = 0;    // Initial index of second subarray
    k = left; // Initial index of merged subarray
    while (i < dim_left && j < dim_right)
    {
        if (temp_left[i] <= temp_right[j])
        {
            data[k] = temp_left[i];
            i++;
        }
        else
        {
            data[k] = temp_right[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of temp_left[], if there are any */
    while (i < dim_left)
    {
        data[k] = temp_left[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of temp_right[], if there are any */
    while (j < dim_right)
    {
        data[k] = temp_right[j];
        j++;
        k++;
    }

    /* Free memory */
    cudaHandleErrorGPU(cudaFree(temp_left));
    cudaHandleErrorGPU(cudaFree(temp_right));
}

__host__ void merge_sort(unsigned short *data, const unsigned long long left, const unsigned long long right)
{
    if (left < right)
    {
        // Same as (l+r)/2, but avoids overflow for large l and h
        unsigned long long mid = left + (right - left) / 2;

        // Sort first half
        merge_sort(data, left, mid);

        // Sort second half
        merge_sort(data, mid + 1, right);

        // Merge the two halves
        merge(data, left, mid, right);
    }
}

__global__ void merge_kernel(unsigned short *data, const unsigned long *offset, const unsigned long total_threads, const unsigned long total_threads_precedent_blocks)
{
    const unsigned long tid = total_threads_precedent_blocks + threadIdx.x;
    unsigned long long start = 0;

    unsigned long long left, mid, right, offset_merge;
    unsigned level_merge = 0, levels_merge = 0;
    unsigned long temp_total_threads = total_threads;
    unsigned thread_to_merge = 0, threads_to_merge = 0;

    unsigned long i;

    // Compute the start for the thread, computing the offset of precedent threads
    for (i = total_threads_precedent_blocks; i < tid; i++)
    {
        start += offset[i];
    }

    /*
        - Log(total_threads)/Log(2) == Log_2(total_threads)
        - Compute number of merge needed in the merge sort
    */
    while (temp_total_threads > 1)
    {
        temp_total_threads /= 2;
        levels_merge++;
    }

    /*
        - Merge the sorted array
    */
    for (level_merge = 0; level_merge <= levels_merge; level_merge++)
    {
        power(2, level_merge, &threads_to_merge);

        if ((tid % threads_to_merge) == 0)
        {
            left = start;
            offset_merge = offset[tid];

            for (thread_to_merge = 1; thread_to_merge < threads_to_merge; thread_to_merge++)
            {
                offset_merge += offset[tid + thread_to_merge];
            }

            right = left + offset_merge - 1;

            if (level_merge == 0)
            {
                mid = left + (right - left) / 2;
            }

            merge_gpu(data, left, mid, right);

            /*
                - Fix since the two merged list are of two different dimension, because the offset is balanced between threads
                - Merge sort expects to have mid as maximum value of the first list
            */
            mid = right;
        }
        __syncthreads();
    }
}

__global__ void merge_blocks_kernel(unsigned short *data, unsigned long long N, ParallelSortConfig config, const unsigned total_threads)
{
    const unsigned tid = threadIdx.x;
    unsigned long long start = 0;

    unsigned long long *block_starting_idx;
    unsigned long long *block_size;
    unsigned long long *thread_offset;
    unsigned long long *thread_mid;
    unsigned total_blocks = total_threads * 2;

    unsigned long long left, mid, right, offset_merge;
    unsigned level_merge = 0, levels_merge = 0;
    unsigned long temp_total_threads = total_threads;
    unsigned thread_to_merge = 0, total_threads_to_merge = 0;

    unsigned i;

    cudaHandleErrorGPU(cudaMalloc((void **)&block_starting_idx, total_blocks * sizeof(unsigned long long)));
    cudaHandleErrorGPU(cudaMalloc((void **)&block_size, total_blocks * sizeof(unsigned long long)));
    cudaHandleErrorGPU(cudaMalloc((void **)&thread_offset, total_threads * sizeof(unsigned long long)));
    cudaHandleErrorGPU(cudaMalloc((void **)&thread_mid, total_threads * sizeof(unsigned long long)));

    /*
        - Compute the start index on the data array for each block
        - Compute the size of the data array to handle for each block
    */
    get_start_index_block(block_starting_idx, N, total_blocks, config.threads_per_block, config.total_threads);
    get_size_block(block_size, block_starting_idx, N, total_blocks, config.threads_per_block, config.total_threads);

    for (unsigned block = 0; block < total_blocks; block++)
    {

        if ((block % 2) == 0)
        {
            // Add the offset of the successive block
            thread_offset[block / 2] = block_size[block] + block_size[block + 1];

            // Compute mid useful during the first level merge
            thread_mid[block / 2] = 0;
            for (i = 0; i <= block; i++)
            {
                thread_mid[block / 2] += block_size[i];
            }
        }
    }

    // Compute new start, end and offset for the thread, computing the offset of precedent threads
    for (unsigned thread = 0; thread < tid; thread++)
    {
        start += thread_offset[thread];
    }

    mid = thread_mid[tid] - 1;

    /*
        - Log(n_threads)/Log(2) == Log_2(n_threads)
        - Compute number of merge needed in the merge sort
    */
    while (temp_total_threads > 1)
    {
        temp_total_threads /= 2;
        levels_merge++;
    }

    /*
        - Merge the sorted array
    */
    for (level_merge = 0; level_merge <= levels_merge; level_merge++)
    {

        power(2, level_merge, &total_threads_to_merge);

        if ((tid % total_threads_to_merge) == 0)
        {
            left = start;
            offset_merge = thread_offset[tid];

            for (thread_to_merge = 1; thread_to_merge < total_threads_to_merge; thread_to_merge++)
            {
                offset_merge += thread_offset[tid + thread_to_merge];
            }

            right = left + offset_merge - 1;

            merge_gpu(data, left, mid, right);

            /*
                - Fix since the two merged list are of two different dimension, because the offset is balanced between threads.
                - Merge sort expects to have mid as maximum value of the first list
            */
            mid = right;
        }
        __syncthreads();
    }

    cudaHandleErrorGPU(cudaFree(block_starting_idx));
    cudaHandleErrorGPU(cudaFree(block_size));
    cudaHandleErrorGPU(cudaFree(thread_offset));
    cudaHandleErrorGPU(cudaFree(thread_mid));
}