#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.cuh"

/* Function to do counting sort of arr[] according to the digit represented by exp. */
__device__ void count_sort(long int *data, unsigned long n, int exp);

/* The main function to that sorts arr[] of size n using Radix Sort */
__device__ void radix_sort(long int *data, unsigned long n);

/* Function to perform only the radix sort*/
__global__ void radix_sort_kernel(long int *data, unsigned long n, unsigned offset, const unsigned long n_threads);