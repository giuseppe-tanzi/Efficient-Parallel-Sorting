#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/utils.cuh"

/* 
    Merge function of the MergeSort Algorithm 
*/
void merge(unsigned short *data, const unsigned long long left, const unsigned long long mid, const unsigned long long right);

/* 
    Merge function of the MergeSort Algorithm to run on GPU
*/
__device__ void merge_gpu(unsigned short *data, const unsigned long long left, const unsigned long long mid, const unsigned long long right);

/* 
    Merge Sort Algorithm 
*/
__host__ void merge_sort(unsigned short *data, const unsigned long long left, const unsigned long long right);