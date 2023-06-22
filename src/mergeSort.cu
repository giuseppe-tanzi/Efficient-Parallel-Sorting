#include "../lib/mergeSort.cuh"

/* Merge function of the MergeSort Algorithm */
__host__ __device__ void merge(unsigned long *data, unsigned long left, unsigned long mid, unsigned long right)
{
    unsigned long i, j, k;
    unsigned long dim_left = mid - left + 1;
    unsigned long dim_right = right - mid;

    /* Create temp arrays */
    unsigned long *temp_left = (unsigned long *)malloc(dim_left * sizeof(unsigned long));
    unsigned long *temp_right = (unsigned long *)malloc(dim_right * sizeof(unsigned long));

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

/* Merge Sort Algorithm */
__host__ void merge_sort(unsigned long*data, unsigned long left, unsigned long right)
{
    if (left < right)
    {
        // Same as (l+r)/2, but avoids overflow for large l and h
        unsigned long mid = left + (right - left) / 2;

        // Sort first half
        merge_sort(data, left, mid);

        // Sort second half
        merge_sort(data, mid + 1, right);

        // Merge the two halves
        merge(data, left, mid, right);
    }
}