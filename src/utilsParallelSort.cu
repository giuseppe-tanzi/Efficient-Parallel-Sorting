#include "../lib/utilsParallelSort.cuh"

ParallelSortConfig determine_config(const unsigned long long N)
{

    ParallelSortConfig config;

    config.partition_size = PARTITION_SIZE;

    if (N <= config.partition_size)
    {
        if (N <= MAXTHREADSPERBLOCK)
        {
            config.total_blocks = 1;
            for (unsigned long long i = N; i >= 2; i--)
            {
                if (is_power_of_two(i))
                {
                    config.total_threads = i;
                    config.partition_size = ceil(N / float(config.total_threads));
                    config.threads_per_block = config.total_threads;
                    break;
                }
            }
        }
        else
        {
            config.threads_per_block = WARPSIZE;
            config.total_threads = WARPSIZE;
            config.total_blocks = 1;
            config.partition_size = ceil(N / (float)config.total_threads);
        }
    }
    else
    {
        config.total_threads = ceil(N / (float)config.partition_size);

        if (config.total_threads <= MAXTHREADSPERBLOCK)
        {
            config.total_blocks = 1;
            if (config.total_threads < WARPSIZE)
            {
                config.total_threads = WARPSIZE;
                config.threads_per_block = WARPSIZE;
            }
            else
            {
                config.threads_per_block = config.total_threads;
            }

            for (unsigned long i = config.total_threads; i >= 2; i--)
            {
                if (is_power_of_two(i))
                {
                    config.total_threads = i;
                    config.partition_size = ceil(N / (float)config.total_threads);
                    config.threads_per_block = config.total_threads;
                    break;
                }
            }
        }
        else
        {
            config.threads_per_block = MAXTHREADSPERBLOCK;
            config.total_blocks = ceil(config.total_threads / (float)config.threads_per_block);

            if (config.total_blocks > MAXBLOCKS)
            {
                config.total_blocks = MAXBLOCKS;
            }

            config.total_threads = (unsigned long)(config.total_blocks * config.threads_per_block);

            for (unsigned long i = config.total_threads; i >= 2; i--)
            {
                config.total_blocks = ceil(i / (float)MAXTHREADSPERBLOCK);
                config.total_threads = (unsigned long)(config.total_blocks * config.threads_per_block);

                if (is_power_of_two(config.total_threads))
                {
                    config.partition_size = ceil(N / (float)config.total_threads);
                    break;
                }
            }
        }
    }

    return config;
}

unsigned long get_n_list_to_merge(unsigned long long N, unsigned long long partition, unsigned long total_threads)
{
    unsigned long thread = 0;
    unsigned long long offset = partition, lists_to_merge = 1;

    for (thread = 1; thread < total_threads; thread++)
    {
        if ((N - offset) > 0)
        {
            offset += ceil((N - offset) / (total_threads - thread));
            lists_to_merge++;
        }
        else
        {
            break;
        }
    }

    return lists_to_merge;
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