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
void init_array(unsigned long *data, unsigned long N)
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
__host__ __device__ void print_array(unsigned long *data, unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
    {
        printf("%li ", data[i]);
    }
    printf("\n");
}

/* 
    Function that checks if the array is ordered 
*/
int check_result(unsigned long *results, unsigned long nitems)
{
    for (unsigned long i = 0; i < nitems - 1; i++)
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

bool IsPowerOfTwo(unsigned long x)
{
    return (x & (x - 1)) == 0;
}

/*
    Function that finds the maximum number in an array
*/
__device__ void get_max(unsigned long *data, unsigned long n, unsigned long *max)
{
    *max = -INFINITY;
    for (int i = 0; i < n; i++)
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