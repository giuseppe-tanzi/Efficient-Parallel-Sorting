#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* Function to get the actual time*/
double gettime(void);

/* Initialize array data */
void init_array(long int *data, unsigned long nitems);

/* Function to print an array */
__host__ __device__ void print_array(long int *data, unsigned long size);

/* Function to check if results is an ordered array */
int check_result(long int *results, unsigned long nitems);

/* Function to check if x is a power of 2*/
bool IsPowerOfTwo(unsigned long x);

/* Function to get maximum value in data; it stores the maximum in max */
__device__ void get_max(long int *data, unsigned long n, long int *max);

/* Function to get the power of base to exp; it stores the result in result*/
__device__ void power(int base, int exp, unsigned *result);
