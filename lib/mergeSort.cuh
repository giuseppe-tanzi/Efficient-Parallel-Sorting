#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Merges two subarrays of data[].
// First subarray is data[l..m]
// Second subarray is data[m+1..r]
__host__ __device__ void merge(unsigned long *data, unsigned long left, unsigned long mid, unsigned long right);

/* Function that performs Merge Sort*/
__host__ void merge_sort(unsigned long *data, unsigned long left, unsigned long right);