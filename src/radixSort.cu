#include "../lib/radixSort.cuh"

__device__ void count_sort_gpu(unsigned short *data, const unsigned long long N, const unsigned exp)
{
    unsigned short *result;
    long long i;
    int count[10] = {0};

    cudaHandleErrorGPU(cudaMalloc((void**)&result, N * sizeof(unsigned short))); 

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

    cudaHandleErrorGPU(cudaFree(result));
}

void count_sort(unsigned short *data, const unsigned long long N, const unsigned exp)
{
    unsigned short *result;
    long long i;
    int count[10] = {0};

    result = (unsigned short *)malloc(N * sizeof(unsigned short));

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

__device__ void radix_sort_gpu(unsigned short *data, const unsigned long long N)
{
    // Find the maximum number to know number of digits
    unsigned short m = 0;
    get_max(data, N, &m);

    // Do counting sort for every digit. Note that instead of passing digit number, exp is passed. 
    // exp is 10^i where i is current digit number
    for (unsigned exp = 1; m / exp > 0; exp *= 10)
    {
        count_sort_gpu(data, N, exp);
    }
}

void radix_sort(unsigned short *data, const unsigned long long N)
{
    // Find the maximum number to know number of digits
    unsigned short m = 0;
    get_max(data, N, &m);

    // Do counting sort for every digit. Note that instead of passing digit number, exp is passed. 
    // exp is 10^i where i is current digit number
    for (unsigned exp = 1; m / exp > 0; exp *= 10)
    {
        count_sort(data, N, exp);
    }
}

__global__ void radix_sort_kernel(unsigned short *data, const unsigned long long N, unsigned long long offset, const unsigned long total_threads)
{
    const unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Variables useful to compute the portion of array for each thread
    unsigned long long start = tid * offset;
    unsigned long long old_offset = 0;
    unsigned long precedent_thread = 0;

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
            for (precedent_thread = 1; precedent_thread < tid; precedent_thread++)
            {
                /*
                    This if-else is useful if there are more thread than needed:
                        - Ensures that no necessary threads remain in idle
                */
                if ((N - old_offset) > 0)
                {
                    // ceil((n - old_offset/total_threads - prec_thread))
                    old_offset += (N - old_offset + (total_threads - precedent_thread) - 1) / (total_threads - precedent_thread);
                }
                else
                {
                    break;
                }
            }
            start = old_offset;
        }

        // ceil((n - start) / (total_threads - tid))
        offset = (N - start + (total_threads - tid) - 1) / (total_threads - tid);
    }

    // More threads than needed
    if ((N - old_offset) > 0) 
    {
        radix_sort_gpu(&data[start], offset);
    }
}