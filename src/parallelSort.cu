#include "../lib/parallelSort.cuh"

ParallelSortConfig determine_config(const unsigned long long N)
{

    ParallelSortConfig config;

    config.partitionSize = PARTITION_SIZE;

    if (N <= config.partitionSize)
    {
        if (N <= MAXTHREADSPERBLOCK)
        {
            config.nBlocks = 1;
            for (unsigned long long i = N; i >= 2; i--)
            {
                if (IsPowerOfTwo(i))
                {
                    config.nTotalThreads = i;
                    config.partitionSize = ceil(N / float(config.nTotalThreads));
                    config.nThreadsPerBlock = config.nTotalThreads;
                    break;
                }
            }
        }
        else
        {
            config.nThreadsPerBlock = WARPSIZE;
            config.nTotalThreads = WARPSIZE;
            config.nBlocks = 1;
            config.partitionSize = ceil(N / (float)config.nTotalThreads);
        }
    }
    else
    {
        config.nTotalThreads = ceil(N / (float)config.partitionSize);

        if (config.nTotalThreads <= MAXTHREADSPERBLOCK)
        {
            config.nBlocks = 1;
            if (config.nTotalThreads < WARPSIZE)
            {
                config.nTotalThreads = WARPSIZE;
                config.nThreadsPerBlock = WARPSIZE;
            }
            else
            {
                config.nThreadsPerBlock = config.nTotalThreads;
            }

            for (unsigned long i = config.nTotalThreads; i >= 2; i--)
            {
                if (IsPowerOfTwo(i))
                {
                    config.nTotalThreads = i;
                    config.partitionSize = ceil(N / (float)config.nTotalThreads);
                    config.nThreadsPerBlock = config.nTotalThreads;
                    break;
                }
            }
        }
        else
        {
            config.nThreadsPerBlock = MAXTHREADSPERBLOCK;
            config.nBlocks = ceil(config.nTotalThreads / (float)config.nThreadsPerBlock);

            if (config.nBlocks > MAXBLOCKS)
            {
                config.nBlocks = MAXBLOCKS;
            }

            config.nTotalThreads = (unsigned long)(config.nBlocks * config.nThreadsPerBlock);

            for (unsigned long i = config.nTotalThreads; i >= 2; i--)
            {
                config.nBlocks = ceil(i / (float)MAXTHREADSPERBLOCK);
                config.nTotalThreads = (unsigned long)(config.nBlocks * config.nThreadsPerBlock);

                if (IsPowerOfTwo(config.nTotalThreads))
                {
                    config.partitionSize = ceil(N / (float)config.nTotalThreads);
                    break;
                }
            }
        }
    }

    return config;
}

__global__ void sort_kernel(unsigned short *data, unsigned long long n, unsigned long long offset, const unsigned long n_threads)
{
    const unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Variables useful to compute the portion of array for each thread
    unsigned long long start = tid * offset;
    unsigned long long end = start + offset - 1;
    unsigned long long old_offset = 0;
    unsigned long prec_thread = 0;

    // Variables useful during the merging phase*
    unsigned long temp_n_threads = n_threads; // Variable useful to compute the numbers of levels during the merging phase
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
                merge_dev(data, left, mid, right);

                /*
                    Merge sort expects to have mid as maximum value of the first list
                    Mid is equal to right to the next level_merge

                */
                mid = right;
            }

            // Needed since the lists to merge to the next level_merge must be ordered
            __syncthreads();
        }
    }
}
