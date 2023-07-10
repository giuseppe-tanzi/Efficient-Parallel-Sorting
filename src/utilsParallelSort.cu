#include "../lib/utilsParallelSort.cuh"

ParallelSortConfig determine_config(const unsigned long long N)
{

    ParallelSortConfig config;

    config.partition_size = PARTITION_SIZE;

    /*
        - N is smaller or equal than the starting partition size of each thread
        - Starting from the maximum number of thread needed (N), it checks that the number of threads is a power of two,
          otherwise the merging phase will not work
    */
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

            if (config.total_threads < WARPSIZE)
            {
                config.threads_per_block = WARPSIZE;
                config.total_threads = WARPSIZE;
                config.total_blocks = 1;
                config.partition_size = ceil(N / (float)config.total_threads);
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
    /*
        - N is greater than the starting partition size of each thread
        - It checks that the number of necessary threads is smaller or equal than the number of threads for each block
          and it computes the partition size
    */
    else
    {
        config.total_threads = ceil(N / (float)config.partition_size);

        /*
            Only one block needed
        */
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

        /*
            More than one block needed
        */
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

    config.required_shared_memory = N * sizeof(unsigned short) / config.total_blocks;
    cudaDeviceGetAttribute(&config.max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);

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
            /*
                Divide the remaining data (N) by the remananing threads
            */
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

__host__ __device__ void get_start_index_block(unsigned long long *block_starting_idx, const unsigned long long N, const unsigned total_blocks, const unsigned long threads_per_block, const unsigned long total_threads)
{
    unsigned long long block = 0;
    unsigned long thread = 0;
    unsigned num_blocks_sort = total_threads / (float)threads_per_block;
    unsigned multiplier = num_blocks_sort / (float)total_blocks;
    unsigned long precedent_threads = multiplier * threads_per_block;

    unsigned long long start = 0;
    unsigned long size = 0;

    for (block = 0; block < total_blocks; block++)
    {
        precedent_threads = multiplier * threads_per_block * block;
        start = 0;
        size = 0;

        if (block == 0)
        {
            start = 0;
        }
        else
        {
            // Compute start index in a recursive way
            for (thread = 0; thread < precedent_threads; thread++)
            {
                if ((N - size) > 0) // More threads than needed
                {
                    /*
                        - Divide the remaining data (N) by the remananing threads
                        - ceil((N - start) / (total_threads - thread))
                    */
                    size = (N - start + (total_threads - thread) - 1) / (total_threads - thread);

                    start += size;
                }
                else
                {
                    break;
                }
            }
        }
        block_starting_idx[block] = start;
    }
}

__host__ __device__ void get_size_block(unsigned long long *block_size, const unsigned long long *block_starting_idx, const unsigned long long N, const unsigned total_blocks, const unsigned long threads_per_block, const unsigned long total_threads)
{
    unsigned long long block = 0;
    unsigned idx_tid = 0; // Actual thread in the block

    unsigned long thread = 0;
    unsigned num_blocks_sort = total_threads / (float)threads_per_block;
    unsigned multiplier = num_blocks_sort / (float)total_blocks;
    unsigned long precedent_threads = multiplier * threads_per_block;

    unsigned long long offset = 0;
    unsigned long size = 0;

    for (block = 0; block < total_blocks; block++)
    {
        precedent_threads = multiplier * threads_per_block * block;
        idx_tid = block * threads_per_block;
        size = block_starting_idx[block];

        // Compute size block in a recursive way
        for (thread = precedent_threads; thread < (block + 1) * threads_per_block * multiplier; thread++)
        {
            if ((N - size) > 0) // More threads than needed
            {
                /*
                    - Divide the remaining data (N) by the remananing threads
                    - ceil((N - size) / (total_threads - thread))
                */
                offset = (N - size + (total_threads - thread) - 1) / (total_threads - thread);
                size += offset;
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

        block_size[block] = size - block_starting_idx[block];
    }
}

__host__ void get_thread_offsets(unsigned long *offsets, const unsigned long long *block_starting_idx, const unsigned long long N, const unsigned total_blocks, const unsigned long threads_per_block, const unsigned long total_threads)
{
    unsigned long long block = 0;
    unsigned idx_tid = 0; // Actual thread in the block

    unsigned long thread = 0;
    unsigned num_blocks_sort = total_threads / (float)threads_per_block;
    unsigned multiplier = num_blocks_sort / (float)total_blocks;
    unsigned long precedent_threads = multiplier * threads_per_block;

    unsigned long size = 0;
    unsigned long long offset = 0;

    // Initialization of the offset for each thread in each block
    for (unsigned i = 0; i < total_blocks * threads_per_block; i++)
    {
        offsets[i] = 0;
    }

    for (block = 0; block < total_blocks; block++)
    {
        precedent_threads = multiplier * threads_per_block * block;
        idx_tid = block * threads_per_block;
        size = block_starting_idx[block];

        // Compute size block in a recursive way
        for (thread = precedent_threads; thread < (block + 1) * threads_per_block * multiplier; thread++)
        {
            if ((N - size) > 0) // More threads than needed
            {
                /*
                    - Divide the remaining data (N) by the remananing threads
                    - ceil((N - size) / (total_threads - thread))
                */
                offset = (N - size + (total_threads - thread) - 1) / (total_threads - thread);
                offsets[idx_tid] += offset;
                size += offset;
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
    }
}