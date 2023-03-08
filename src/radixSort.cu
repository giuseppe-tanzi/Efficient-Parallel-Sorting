#include "../lib/radixSort.cuh"

__device__ void count_sort(long int *data, unsigned long n, int exp)
{
    long int *result = (long int *)malloc(n * sizeof(long int)); // output array
    long int i, count[10] = {0};

    // Store count of occurrences in count[]
    for (i = 0; i < n; i++)
    {
        count[(data[i] / exp) % 10]++;
    }

    // Change count[i] so that count[i] now contains actual position of this digit in output[]
    for (i = 1; i < 10; i++)
    {
        count[i] += count[i - 1];
    }

    // Build the result array
    for (i = n - 1; i >= 0; i--)
    {
        result[count[(data[i] / exp) % 10] - 1] = data[i];
        count[(data[i] / exp) % 10]--;
    }

    // Copy the output array to data[], so that data[] now contains sorted numbers according to current digit
    for (i = 0; i < n; i++)
    {
        data[i] = result[i];
    }

    free(result);
}

__device__ void radix_sort(long int *data, unsigned long n)
{
    // Find the maximum number to know number of digits
    long int m = 0;
    get_max(data, n, &m);

    // Do counting sort for every digit. Note that instead of passing digit number, exp is passed. 
    // exp is 10^i where i is current digit number
    for (int exp = 1; m / exp > 0; exp *= 10)
    {
        count_sort(data, n, exp);
    }
}