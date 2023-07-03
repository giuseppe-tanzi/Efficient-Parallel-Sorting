#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Merges two subarrays of data[].
// First subarray is data[l..m]
// Second subarray is data[m+1..r]
__host__ __device__ void merge(unsigned short *data, const unsigned long long left, const unsigned long long mid, const unsigned long long right);

/* Function that performs Merge Sort*/
__host__ void merge_sort(unsigned short *data, const unsigned long long left, const unsigned long long right);