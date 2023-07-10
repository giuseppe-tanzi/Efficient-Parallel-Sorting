#include "../lib/parallelSort.cuh"

bool parallel_sort(unsigned short *dev_a,
                   const unsigned long long N,
                   ParallelSortConfig config,
                   const size_t size_blocks,
                   const unsigned blocks_involved_in_merging,
                   unsigned long long *block_starting_idx,
                   unsigned long long *block_size,
                   unsigned long *thread_offset,
                   unsigned long *dev_thread_offset,
                   bool shared_memory)
{

    bool used_shared_memory = false;
    const size_t required_shared_memory_data = N * sizeof(unsigned long long);

    if (config.total_blocks == 1)
    {
        /*
            Compute the radix sort phase and the merging sort phase in the same kernel
        */
        if ((config.required_shared_memory <= config.max_shared_memory_per_block) && shared_memory == true)
        {
            used_shared_memory = true;
            // Launch the kernel with the correct shared memory size
            sort_kernel_shared<<<config.gridSize, config.blockSize, config.required_shared_memory>>>(dev_a, N, config.partition_size, config.total_threads);
        }
        else
        {
            sort_kernel<<<config.gridSize, config.blockSize>>>(dev_a, N, config.partition_size, config.total_threads);
        }
    }
    else
    {
        /*
            Compute the sorting in two different phase
        */

        /*
            The radix sort is computed on the entire array with the all necessary blocks
        */

        if ((config.required_shared_memory <= config.max_shared_memory_per_block) && shared_memory == true)
        {
            used_shared_memory = true;

            // Launch the kernel with the correct shared memory size
            radix_sort_kernel_shared<<<config.gridSize, config.blockSize, config.required_shared_memory>>>(dev_a, N, config.partition_size, config.total_threads);
        }
        else
        {
            radix_sort_kernel<<<config.gridSize, config.blockSize>>>(dev_a, N, config.partition_size, config.total_threads);
        }

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
            - The merging phase is computed using a different number of blocks, since the number of necessary threads is smaller
            - By doing so all the threads in each block performs a merge during the first level of the merging phase
            - Then, the sorting is called on only one block in order to sort all the portion of array sorted by each block
        */
        /*
            - It calls the merge kernel on each block
            - Each block has a defined portion of the array to handle and a precise number of lists to merge
            - The array will have blocks_involved_in_merging lists to merge at the end of the for-loop
        */
        for (unsigned block = 0; block < blocks_involved_in_merging; block++)
        {
            merge_kernel<<<1, config.blockSize>>>(dev_a + block_starting_idx[block], dev_thread_offset, config.threads_per_block, block * config.threads_per_block);
        }

        /*
            It performs the last merging phase with only one block since the number of lists to sort are surely
            less than the maximum thread number for each block
        */
        if (blocks_involved_in_merging > 1)
        {
            if ((required_shared_memory_data <= config.max_shared_memory_per_block) && shared_memory == true)
            {
                used_shared_memory = true;

                // Launch the kernel with the correct shared memory size
                merge_blocks_kernel_shared<<<1, blocks_involved_in_merging / 2, required_shared_memory_data>>>(dev_a, N, config, blocks_involved_in_merging / 2);
            }
            else
            {
                merge_blocks_kernel<<<1, blocks_involved_in_merging / 2>>>(dev_a, N, config, blocks_involved_in_merging / 2);
            }
        }
    }

    return used_shared_memory;
}

__global__ void sort_kernel(unsigned short *data, const unsigned long long N, unsigned long long offset, const unsigned long total_threads)
{
    const unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Variables useful to compute the portion of array for each thread
    unsigned long long start = tid * offset;
    unsigned long long end = start + offset - 1;
    unsigned long long old_offset = 0;
    unsigned long precedent_thread = 0;

    // Variables useful during the merging phase*
    unsigned long temp_total_threads = total_threads; // Variable useful to compute the numbers of levels during the merging phase
    unsigned level_merge = 0, levels_merge = 0, threads_to_merge = 0;
    unsigned long long offset_merge = 0;
    unsigned long long left = 0, mid = 0, right = 0;

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
            for (precedent_thread = 1; precedent_thread < tid; precedent_thread++)
            {
                /*
                    This if-else is useful if there are more thread than needed:
                        - Ensures that no necessary thread remain in idle
                */
                if ((N - old_offset) > 0)
                {
                    /*
                        - Compute old offset by computing how much offset the precedent threads have
                        - ceil((N - old_offset) / (total_threads - precedent_thread))
                    */
                    old_offset += (N - old_offset + (total_threads - precedent_thread) - 1) / (total_threads - precedent_thread);
                }
                else
                {
                    break;
                }
            }
            start = old_offset;
        }

        /*
            ceil((N - start) / (total_threads - tid))
        */
        offset = (N - start + (total_threads - tid) - 1) / (total_threads - tid);
        end = start + offset - 1;
    }

    /*
        This if-else is useful if there are more thread than needed:
            - It ensures that no necessary thread remain in idle
    */
    if ((N - old_offset) > 0)
    {

        /*
            Log(num_threads)/Log(2) == Log_2(num_threads)
            Compute number of merge needed in the merge sort
        */
        while (temp_total_threads > 1)
        {
            temp_total_threads /= 2;
            levels_merge++;
        }

        radix_sort_gpu(&data[start], offset);
        __syncthreads();

        // Merge - Phase
        for (level_merge = 1; level_merge <= levels_merge; level_merge++)
        {
            /*
                - At first level, mid is equal to the end of the portion sorted by the thread since during the merging phase,
                  mid is the final index of the left portion.
            */
            if (level_merge == 1)
            {
                mid = end;
            }

            /*
                - Threads_to_merge = 2^(level_merge) - Useful to exclude no necessary thread in the successive level
                - Threads_to_merge is equal to the number of threads merged from the first level of the merging phase
            */
            power(2, level_merge, &threads_to_merge);

            if ((tid % threads_to_merge) == 0)
            {
                left = start;
                offset_merge = offset;

                /*
                    Useful to compute the size of the resulting list after the current level_merge
                */
                for (precedent_thread = tid + 1; precedent_thread < tid + threads_to_merge; precedent_thread++)
                {
                    /*
                        - Compute offset_merge by computing how much offset the precedent threads have
                        - ceil((N - start - offset_merge) / (total_threads - precedent_thread))
                    */
                    offset_merge += (N - start - offset_merge + (total_threads - precedent_thread) - 1) / (total_threads - precedent_thread);
                }

                right = left + offset_merge - 1;

                merge_gpu(data, left, mid, right);

                /*
                    - Merge sort expects to have mid as maximum value of the first list
                    - Mid is equal to right to the next level_merge
                */
                mid = right;
            }

            // Needed since the lists to merge to the next level_merge must be ordered
            __syncthreads();
        }
    }
}

__global__ void sort_kernel_shared(unsigned short *data, const unsigned long long N, unsigned long long offset, const unsigned long total_threads)
{
    extern __shared__ unsigned short shared_data[];

    const unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Variables useful to compute the portion of array for each thread
    unsigned long long start = tid * offset;
    unsigned long long end = start + offset - 1;
    unsigned long long old_offset = 0;
    unsigned long precedent_thread = 0;

    // Variables useful during the merging phase*
    unsigned long temp_total_threads = total_threads; // Variable useful to compute the numbers of levels during the merging phase
    unsigned level_merge = 0, levels_merge = 0, threads_to_merge = 0;
    unsigned long long offset_merge = 0;
    unsigned long long left = 0, mid = 0, right = 0;

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
            for (precedent_thread = 1; precedent_thread < tid; precedent_thread++)
            {
                /*
                    This if-else is useful if there are more thread than needed:
                        - Ensures that no necessary thread remain in idle
                */
                if ((N - old_offset) > 0)
                {
                    /*
                        - Compute old offset by computing how much offset the precedent threads have
                        - ceil((N - old_offset) / (total_threads - precedent_thread))
                    */
                    old_offset += (N - old_offset + (total_threads - precedent_thread) - 1) / (total_threads - precedent_thread);
                }
                else
                {
                    break;
                }
            }
            start = old_offset;
        }

        /*
            - ceil((N - start) / (total_threads - tid))
        */
        offset = (N - start + (total_threads - tid) - 1) / (total_threads - tid);
        end = start + offset - 1;
    }

    /*
        Copy the data to the shared memory
    */
    for (unsigned long long i = start; i <= end && i < N; i++)
    {
        shared_data[i] = data[i];
    }
    __syncthreads();

    /*
        This if-else is useful if there are more thread than needed:
            - It ensures that no necessary thread remain in idle
    */
    if ((N - old_offset) > 0)
    {

        /*
            Log(num_threads)/Log(2) == Log_2(num_threads)
            Compute number of merge needed in the merge sort
        */
        while (temp_total_threads > 1)
        {
            temp_total_threads /= 2;
            levels_merge++;
        }

        radix_sort_gpu(&shared_data[start], offset);
        __syncthreads();

        // Merge - Phase
        for (level_merge = 1; level_merge <= levels_merge; level_merge++)
        {
            /*
                - At first level, mid is equal to the end of the portion sorted by the thread since during the merging phase,
                  mid is the final index of the left portion.
            */
            if (level_merge == 1)
            {
                mid = end;
            }

            /*
                - Threads_to_merge = 2^(level_merge) - Useful to exclude no necessary thread in the successive level
                - Threads_to_merge is equal to the number of threads merged from the first level of the merging phase
            */
            power(2, level_merge, &threads_to_merge);

            if ((tid % threads_to_merge) == 0)
            {
                left = start;
                offset_merge = offset;

                /*
                    Useful to compute the size of the resulting list after the current level_merge
                */
                for (precedent_thread = tid + 1; precedent_thread < tid + threads_to_merge; precedent_thread++)
                {
                    /*
                        - Compute offset_merge by computing how much offset the precedent threads have
                        - ceil((N - start - offset_merge) / (total_threads - precedent_thread))
                    */
                    offset_merge += (N - start - offset_merge + (total_threads - precedent_thread) - 1) / (total_threads - precedent_thread);
                }

                right = left + offset_merge - 1;

                merge_gpu(shared_data, left, mid, right);

                /*
                    - Merge sort expects to have mid as maximum value of the first list
                    - Mid is equal to right to the next level_merge
                */
                mid = right;
            }

            // Needed since the lists to merge to the next level_merge must be ordered
            __syncthreads();
        }
    }

    /*
        Copy the data back to the global memory
    */
    for (unsigned long long i = start; i <= end && i < N; i++)
    {
        data[i] = shared_data[i];
    }

    __syncthreads();
}