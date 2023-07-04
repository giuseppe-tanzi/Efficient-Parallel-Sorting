#include "../lib/utilsParallelSort.cuh"

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

unsigned long get_n_list_to_merge(unsigned long long N, unsigned long long partition, unsigned long num_threads)
{
    unsigned long thread = 0;
    unsigned long long offset = partition, n_list_to_merge = 1;

    for (thread = 1; thread < num_threads; thread++)
    {
        if ((N - offset) > 0)
        {
            offset += ceil((N - offset) / (num_threads - thread));
            n_list_to_merge++;
        }
        else
        {
            break;
        }
    }

    return n_list_to_merge;
}

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
            // Compute start index in a recursive way
            for (thread = 0; thread < precedent_threads; thread++)
            {
                if ((N - size_v) > 0) // More threads than needed
                {
                    // ceil((N - start_v) / (total_threads - thread))
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

        // Compute size block in a recursive way
        for (thread = precedent_threads; thread < (current_block + 1) * MAXTHREADSPERBLOCK * multiplier; thread++)
        {
            if ((N - size_v) > 0) // More threads than needed
            {
                // ceil((N - size_v) / (total_threads - thread))
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