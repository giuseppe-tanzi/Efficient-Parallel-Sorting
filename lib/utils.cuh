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

/*
    Function that returns the current time
*/
double gettime(void);

/*
    Function that randomly initializes an array from MIN VALUE to MAX VALUE
*/
void init_array(unsigned short *data, const unsigned long long N);

/*
    Function that prints an array
*/
__host__ __device__ void print_array(const unsigned short *data, const unsigned long long N);

/*
    Function that checks if the array is ordered
*/
int check_result(unsigned short *results, const unsigned long long N);

/* Function to check if x is a power of 2*/
bool IsPowerOfTwo(const unsigned long x);

/*
    Function that finds the maximum number in an array
*/
__device__ void get_max(unsigned short *data, const unsigned long long N, unsigned short *max);

/*
    Function useful to compute the base to the power of exp
*/
__device__ void power(unsigned base, unsigned exp, unsigned *result);
