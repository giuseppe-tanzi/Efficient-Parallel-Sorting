#include "../lib/utils.cuh"

/*
    Function that returns the current time
*/
double gettime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

/*
    Function that randomly initializes an array from 0 to N
*/
void init_array(unsigned long *data, const unsigned long N)
{
    long int temp;
    srand(42); // Ensure the determinism

    for (unsigned long i = 0; i < N; i++)
    {
        data[i] = N - 1 - i;
    }

    /* Random shuffle */
    for (unsigned long i = 0; i < N - 1; i++)
    {
        size_t j = i + rand() / (RAND_MAX / (N - i) + 1);
        temp = data[j];
        data[j] = data[i];
        data[i] = temp;
    }
}

/*
    Function that prints an array
*/
__host__ __device__ void print_array(const unsigned long *data, const unsigned long N)
{
    for (unsigned long i = 0; i < N; i++)
    {
        printf("%li ", data[i]);
    }
    printf("\n");
}

/*
    Function that checks if the array is ordered
*/
int check_result(unsigned long *results, const unsigned long N)
{
    for (unsigned long i = 0; i < N - 1; i++)
    {
        if (results[i] > results[i + 1])
        {
            printf("Check failed: data[%lu] = %li, data[%lu] = %li\n", i, results[i], i + 1, results[i + 1]);
            printf("%li is greater than %li\n", results[i], results[i + 1]);
            return 0;
        }
    }
    printf("Check OK\n");
    return 1;
}

bool IsPowerOfTwo(const unsigned long x)
{
    return (x & (x - 1)) == 0;
}

/*
    Function that finds the maximum number in an array
*/
__device__ void get_max(unsigned long *data, const unsigned long N, unsigned long *max)
{
    *max = -INFINITY;
    for (int i = 0; i < N; i++)
    {
        if (data[i] > *max)
        {
            *max = data[i];
        }
    }
}

/*
    Function useful to compute the base to the power of exp
*/
__device__ void power(int base, int exp, unsigned *result)
{
    *result = 1;
    for (;;)
    {
        if (exp & 1)
            *result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }
}

void determine_config(const unsigned long N, unsigned long *n_threads_per_block, unsigned long *n_blocks,
                      unsigned long *n_total_threads, unsigned long *partition_size)
{

    *partition_size = PARTITION_SIZE;

    if (N <= *partition_size)
    {
        if (N <= MAXTHREADSPERBLOCK)
        {
            *n_blocks = 1;
            for (unsigned long i = N; i >= 2; i--)
            {
                if (IsPowerOfTwo(i))
                {
                    *n_total_threads = i;
                    *partition_size = ceil(N / float(*n_total_threads));
                    *n_threads_per_block = *n_total_threads;
                    break;
                }
            }
        }
        else
        {
            *n_threads_per_block = WARPSIZE;
            *n_total_threads = WARPSIZE;
            *n_blocks = 1;
            *partition_size = ceil(N / (float)*n_total_threads);
        }
    }
    else
    {
        *n_total_threads = ceil(N / (float)*partition_size);

        if (*n_total_threads <= MAXTHREADSPERBLOCK)
        {
            *n_blocks = 1;
            if (*n_total_threads < WARPSIZE)
            {
                *n_total_threads = WARPSIZE;
                *n_threads_per_block = WARPSIZE;
            }
            else
            {
                *n_threads_per_block = *n_total_threads;
            }

            for (unsigned long i = *n_total_threads; i >= 2; i--)
            {
                if (IsPowerOfTwo(i))
                {
                    *n_total_threads = i;
                    *partition_size = ceil(N / (float)*n_total_threads);
                    *n_threads_per_block = *n_total_threads;
                    break;
                }
            }
        }
        else
        {
            *n_threads_per_block = MAXTHREADSPERBLOCK;
            *n_blocks = ceil(*n_total_threads / (float)*n_threads_per_block);

            if (*n_blocks > MAXBLOCKS)
            {
                *n_blocks = MAXBLOCKS;
            }

            *n_total_threads = (unsigned long)(*n_blocks * *n_threads_per_block);

            for (unsigned long i = *n_total_threads; i >= 2; i--)
            {
                *n_blocks = ceil(i / (float)MAXTHREADSPERBLOCK);
                *n_total_threads = (unsigned long)(*n_blocks * *n_threads_per_block);

                if (IsPowerOfTwo(*n_total_threads))
                {
                    *partition_size = ceil(N / (float)*n_total_threads);
                    break;
                }
            }
        }
    }
}
