#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"

/*
    Useful to check errors in the cuda kernels on CPU
*/
__host__ void gpuAssert(cudaError_t code, const char *file, int line, bool abort);

/*
    Useful to check errors in the cuda kernels on GPU
*/
__device__ void gpuAssert_gpu(cudaError_t code, const char *file, int line, bool abort);

/*
    Useful to check errors in the cuda kernels on CPU
*/
#define cudaHandleError(ans) gpuAssert((ans), __FILE__, __LINE__, true)

/*
    Useful to check errors in the cuda kernels on GPU
*/
#define cudaHandleErrorGPU(ans) gpuAssert_gpu((ans), __FILE__, __LINE__, true)

/*
    Function that returns the current time
*/
double get_time(void);

/*
    Function that randomly initializes an array from MIN VALUE to MAX VALUE
*/
void init_array(unsigned short *data, const unsigned long long N);

/*
    Function that prints an array
*/
__host__ void print_array(const unsigned short *data, const unsigned long long N);

/*
    Function that checks if the array is sorted
*/
bool is_sorted(unsigned short *result, const unsigned long long N);

/* 
    Function to check if x is a power of 2
*/
bool is_power_of_two(const unsigned long x);

/*
    Function that finds the maximum number in an array
*/
__host__ __device__ void get_max(unsigned short *data, const unsigned long long N, unsigned short *max);

/*
    Function useful to compute the base to the power of exp
*/
__device__ void power(unsigned base, unsigned exp, unsigned *result);

/*
    Function that prints the statistics of the algorithms
*/
void print_table(int n_algorithms, char algorithms[][100], char machine[][4], unsigned long threads[], bool used_shared[], bool correctness[], double elapsed_time[]);
