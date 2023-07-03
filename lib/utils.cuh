#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort);

/*
    Useful to check errors in the cuda kernels
*/
#define cudaHandleError(ans) gpuAssert((ans), __FILE__, __LINE__, true)

/* Function to get the actual time*/
double gettime(void);

/* Initialize array data */
void init_array(unsigned short *data, const unsigned long long N);

/* Function to print an array */
__host__ __device__ void print_array(const unsigned short *data, const unsigned long long N);

/* Function to check if results is an ordered array */
int check_result(unsigned short *results, const unsigned long long N);

/* Function to check if x is a power of 2*/
bool IsPowerOfTwo(const unsigned long x);

/* Function to get maximum value in data; it stores the maximum in max */
__device__ void get_max(unsigned short *data, const unsigned long long N, unsigned short *max);

/* Function to get the power of base to exp; it stores the result in result*/
__device__ void power(unsigned base, unsigned exp, unsigned long *result);
