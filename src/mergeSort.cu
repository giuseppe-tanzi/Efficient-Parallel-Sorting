#include "../lib/mergeSort.cuh"

void merge(unsigned short *data, const unsigned long long left, const unsigned long long mid, const unsigned long long right)
{
    unsigned long long i, j, k;
    unsigned long long dim_left = mid - left + 1;
    unsigned long long dim_right = right - mid;

    /* Create temp arrays */
    unsigned short *temp_left = (unsigned short *)malloc(dim_left * sizeof(unsigned short));
    unsigned short *temp_right = (unsigned short *)malloc(dim_right * sizeof(unsigned short));

    /* Copy data to temp arrays temp_left[] and temp_right[] */
    for (i = 0; i < dim_left; i++)
    {
        temp_left[i] = data[left + i];
    }

    for (j = 0; j < dim_right; j++)
    {
        temp_right[j] = data[mid + 1 + j];
    }

    /* Merge the temp arrays back into data[l..r]*/
    i = 0;    // Initial index of first subarray
    j = 0;    // Initial index of second subarray
    k = left; // Initial index of merged subarray
    while (i < dim_left && j < dim_right)
    {
        if (temp_left[i] <= temp_right[j])
        {
            data[k] = temp_left[i];
            i++;
        }
        else
        {
            data[k] = temp_right[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of temp_left[], if there are any */
    while (i < dim_left)
    {
        data[k] = temp_left[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of temp_right[], if there are any */
    while (j < dim_right)
    {
        data[k] = temp_right[j];
        j++;
        k++;
    }

    /* Free memory */
    free(temp_left);
    free(temp_right);
}

__device__ void merge_gpu(unsigned short *data, const unsigned long long left, const unsigned long long mid, const unsigned long long right)
{
    unsigned long long i, j, k;
    unsigned long long dim_left = mid - left + 1;
    unsigned long long dim_right = right - mid;
    unsigned short *temp_left;
    unsigned short *temp_right;

    /* Create temp arrays */
    cudaHandleErrorGPU(cudaMalloc((void**)&temp_left, dim_left * sizeof(unsigned short))); 
    cudaHandleErrorGPU(cudaMalloc((void**)&temp_right, dim_right * sizeof(unsigned short)));
    
    /* Copy data to temp arrays temp_left[] and temp_right[] */
    for (i = 0; i < dim_left; i++)
    {
        temp_left[i] = data[left + i];
    }

    for (j = 0; j < dim_right; j++)
    {
        temp_right[j] = data[mid + 1 + j];
    }

    /* Merge the temp arrays back into data[l..r]*/
    i = 0;    // Initial index of first subarray
    j = 0;    // Initial index of second subarray
    k = left; // Initial index of merged subarray
    while (i < dim_left && j < dim_right)
    {
        if (temp_left[i] <= temp_right[j])
        {
            data[k] = temp_left[i];
            i++;
        }
        else
        {
            data[k] = temp_right[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of temp_left[], if there are any */
    while (i < dim_left)
    {
        data[k] = temp_left[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of temp_right[], if there are any */
    while (j < dim_right)
    {
        data[k] = temp_right[j];
        j++;
        k++;
    }

    /* Free memory */
    cudaHandleErrorGPU(cudaFree(temp_left));
    cudaHandleErrorGPU(cudaFree(temp_right));
}

__host__ void merge_sort(unsigned short *data, const unsigned long long left, const unsigned long long right)
{
    if (left < right)
    {
        // Same as (l+r)/2, but avoids overflow for large l and h
        unsigned long long mid = left + (right - left) / 2;

        // Sort first half
        merge_sort(data, left, mid);

        // Sort second half
        merge_sort(data, mid + 1, right);

        // Merge the two halves
        merge(data, left, mid, right);
    }
}