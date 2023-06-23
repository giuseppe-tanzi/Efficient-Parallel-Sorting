#include "../lib/radixSort.cuh"

__device__ void count_sort(unsigned long *data, const unsigned long N, const int exp)
{
    long int *result = (long int *)malloc(N * sizeof(long int)); // output array
    long int i, count[10] = {0};

    // Store count of occurrences in count[]
    for (i = 0; i < N; i++)
    {
        count[(data[i] / exp) % 10]++;
    }

    // Change count[i] so that count[i] now contains actual position of this digit in output[]
    for (i = 1; i < 10; i++)
    {
        count[i] += count[i - 1];
    }

    // Build the result array
    for (i = N - 1; i >= 0; i--)
    {
        result[count[(data[i] / exp) % 10] - 1] = data[i];
        count[(data[i] / exp) % 10]--;
    }

    // Copy the output array to data[], so that data[] now contains sorted numbers according to current digit
    for (i = 0; i < N; i++)
    {
        data[i] = result[i];
    }

    free(result);
}

__device__ void radix_sort(unsigned long *data, const unsigned long N)
{
    // Find the maximum number to know number of digits
    unsigned long m = 0;
    get_max(data, N, &m);

    // Do counting sort for every digit. Note that instead of passing digit number, exp is passed. 
    // exp is 10^i where i is current digit number
    for (int exp = 1; m / exp > 0; exp *= 10)
    {
        count_sort(data, N, exp);
    }
}

__global__ void radix_sort_kernel(unsigned long *data, const unsigned long N, unsigned offset, const unsigned long n_threads)
{
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Variables useful to compute the portion of array for each thread
    unsigned long start = tid * offset;
    unsigned old_offset = 0;
    unsigned prec_thread = 0;

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
            start = 0;
            old_offset = offset;
            for (prec_thread = 1; prec_thread < tid; prec_thread++)
            {
                /*
                    This if-else is useful if there are more thread than needed:
                        - Ensures that no necessary thread remain in idle
                */
                if ((N - old_offset) > 0) // MORE THREAD THAN NEEDED
                {
                    // ceil((n - old_offset/n_threads - prec_thread))
                    old_offset += (N - old_offset + (n_threads - prec_thread) - 1) / (n_threads - prec_thread);
                }
                else
                {
                    break;
                }
            }
            start = old_offset;
        }

        // ceil((n - start) / (n_threads - tid))
        offset = (N - start + (n_threads - tid) - 1) / (n_threads - tid);
    }

    if ((N - old_offset) > 0) // MORE THREAD THAN NEEDED
    {
        radix_sort(&data[start], offset);
    }
}