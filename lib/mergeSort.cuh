#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* 
    Merge function of the MergeSort Algorithm 
*/
__host__ __device__ void merge(unsigned short *data, const unsigned long long left, const unsigned long long mid, const unsigned long long right);

/* 
    Merge Sort Algorithm 
*/
__host__ void merge_sort(unsigned short *data, const unsigned long long left, const unsigned long long right);