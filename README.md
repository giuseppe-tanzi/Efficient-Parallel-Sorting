# Parallel-Sorting

Project work for the 1st Module of "Architectures and Platforms for Artificial Intelligence" course of the Artificial Intelligence Master's Degree at University of Bologna.

## Overview 
The existing sequential sorting algorithms, such as bubble sort, quick sort, radix sort, and merge
sort, follow step-by-step procedures that perform comparisons and swaps to gradually organize the
data. While these algorithms are conceptually simple to implement, their performance tends to
degrade with larger input sizes.
Parallel sorting algorithms, on the other hand, aim to overcome these limitations by dividing
the dataset into smaller portions that are processed independently in parallel. This approach
allows parallel algorithms to exploit the massive parallelism offered by modern GPUs, resulting in
significant performance improvements for sorting operations.


In this repository there is the implementation of a parallel sorting algorithm in CUDA C.

## Algorithm

I have developed a solution that combines the principles of Radix Sort and Merge Sort. The algorithm
leverages the advantages of Radix Sort’s non-comparative nature and Merge Sort’s divideand-
conquer strategy to achieve efficient parallel sorting. My solution builds upon the efficiency of Radix Sort by utilizing it as the base sorting algorithm
within Merge Sort.

Sure! Here's the usage section you can add to your README:

## Usage

To compile the program, you need to have the NVIDIA CUDA Toolkit installed on your system. If you haven't installed it yet, please refer to the official documentation for installation instructions.

Once you have the CUDA Toolkit set up, follow these steps to compile and run the program:

1. Clone the repository to your local machine:
2. Navigate to the project directory:
3. Compile the program using the provided Makefile:
   ```
   make
   ```

   This command will compile the CUDA code and generate an executable file named `main.out`.

4. Run the program with the desired arguments. The program requires the size of the array as a command-line argument. For example, to sort an array of size 10000, run:
   ```
   ./main.out 10000
   ```

   Replace `10000` with the desired size of the array.

5. Optionally, you can include the `-w` argument to write the statistics of the sorting operation to a CSV file. This file will contain information such as execution time and other parameters. To enable writing statistics to a CSV file, use the following command:
   ```
   ./main.out <array_size> -w
   ```

   Replace `<array_size>` with the desired size of the array.

   Note: If you omit the `-w` argument, the program will only display the statistics on the console without writing them to a file.
