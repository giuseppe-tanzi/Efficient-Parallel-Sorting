# Efficient Parallel Sorting Algorithm in CUDA

This project focuses on the implementation of an efficient parallel sorting algorithm in CUDA C. The conventional sequential sorting algorithms, including bubble sort, quick sort, radix sort, and merge sort, exhibit performance degradation with larger input sizes. To address this issue, parallel sorting algorithms are employed, dividing the dataset into smaller portions processed independently in parallel. This project leverages the massive parallelism offered by modern GPUs, resulting in significant performance improvements for sorting operations.

## Algorithm

The implemented solution combines the principles of Radix Sort and Merge Sort. This algorithm capitalizes on Radix Sort's non-comparative nature and Merge Sort's divide-and-conquer strategy to achieve efficient parallel sorting. The approach builds upon the efficiency of Radix Sort by utilizing it as the base sorting algorithm within Merge Sort.

## Usage

To compile and run the program:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Compile the program using the provided Makefile:
   ```
   make
   ```
   This command generates an executable file named `main.out`.

4. Run the program with the desired arguments. The program requires the size of the array as a command-line argument. For example:
   ```
   ./main.out 10000
   ```
   Replace `10000` with the desired size of the array.

5. Optionally, include the `-w` argument to write sorting operation statistics to a CSV file. To enable this feature:
   ```
   ./main.out <array_size> -w
   ```
   Replace `<array_size>` with the desired size of the array.

   Note: Omitting the `-w` argument will display the statistics on the console without writing them to a file.

Ensure you have the NVIDIA CUDA Toolkit installed before compiling the program. Refer to the official documentation for installation instructions.
