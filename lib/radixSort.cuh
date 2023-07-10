#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.cuh"

/*
    Function to do counting sort of the data according to the digit represented by exp on GPU
*/
__device__ void count_sort_gpu(unsigned short *data, const unsigned long long N, const unsigned exp);

/*
    Function to do counting sort of the data according to the digit represented by exp on CPU
*/
void count_sort(unsigned short *data, const unsigned long long N, const unsigned exp);

/*
    The main function to that sorts the data of size N using Radix Sort on GPU
*/
__device__ void radix_sort_gpu(unsigned short *data, const unsigned long long N);

/*
    The main function to that sorts the data of size N using Radix Sort on CPU
*/
void radix_sort(unsigned short *data, const unsigned long long N);

/*
    Function to perform only the Radix Sort
*/
__global__ void radix_sort_kernel(unsigned short *data, const unsigned long long N, unsigned long long offset, const unsigned long total_threads);

/*
    Function to perform only the Radix Sort using the shared memory
*/
__global__ void radix_sort_kernel_shared(unsigned short *data, const unsigned long long N, unsigned long long offset, const unsigned long total_threads);