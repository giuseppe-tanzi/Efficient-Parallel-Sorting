#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"

/* Function to get the actual time*/
double gettime(void);

/* Initialize array data */
void init_array(unsigned long *data, const unsigned long N);

/* Function to print an array */
__host__ __device__ void print_array(const unsigned long *data, const unsigned long N);

/* Function to check if results is an ordered array */
int check_result(unsigned long *results, const unsigned long N);

/* Function to check if x is a power of 2*/
bool IsPowerOfTwo(const unsigned long x);

/* Function to get maximum value in data; it stores the maximum in max */
__device__ void get_max(unsigned long *data, const unsigned long N, unsigned long *max);

/* Function to get the power of base to exp; it stores the result in result*/
__device__ void power(int base, int exp, unsigned *result);

void determine_config(const unsigned long N, unsigned long *n_threads_per_block, unsigned long *n_blocks,
                      unsigned long *n_total_threads, unsigned long *partition_size);
