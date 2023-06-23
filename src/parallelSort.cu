#include "../lib/parallelSort.cuh"

ParallelSortConfig determine_config(const unsigned long N)
{

    ParallelSortConfig config;

    config.partitionSize = PARTITION_SIZE;

    if (N <= config.partitionSize)
    {
        if (N <= MAXTHREADSPERBLOCK)
        {
            config.nBlocks = 1;
            for (unsigned long i = N; i >= 2; i--)
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
