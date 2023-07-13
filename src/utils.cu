#include "../lib/utils.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUerror: %s\nCode: %d\nFile: %s\nLine: %d\n", cudaGetErrorString(code), code, file, line);
        if (abort)
            exit(code);
    }
}

__device__ void gpuAssert_gpu(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        const char *errorString = cudaGetErrorString(code);

        printf("GPUerror: %s\nCode: %d\nFile: %s\nLine: %d\n", errorString, code, file, line);

        if (abort)
            asm("trap;");
    }
}

double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

void init_array(unsigned short *data, const unsigned long long N)
{
    srand(42); // Ensure the determinism

    for (unsigned long long i = 0; i < N; i++)
    {
        data[i] = rand() % (MAX_VALUE - MIN_VALUE + 1) + MIN_VALUE;
        ;
    }
}

__host__ void print_array(const unsigned short *data, const unsigned long long N)
{
    for (unsigned long long i = 0; i < N; i++)
    {
        printf("%hu ", data[i]);
    }
    printf("\n");
}

bool is_sorted(unsigned short *result, const unsigned long long N)
{
    for (unsigned long long i = 0; i < N - 1; i++)
    {
        if (result[i] > result[i + 1])
        {
            return false;
        }
    }
    return true;
}

bool is_power_of_two(const unsigned long x)
{
    return (x & (x - 1)) == 0;
}

__host__ __device__ void get_max(unsigned short *data, const unsigned long long N, unsigned short *max)
{
    *max = 0;
    for (unsigned long long i = 0; i < N; i++)
    {
        if (data[i] > *max)
        {
            *max = data[i];
        }
    }
}

__device__ void power(unsigned base, unsigned exp, unsigned *result)
{
    *result = 1;
    for (;;)
    {
        if (exp & 1)
            *result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }
}

void print_table(int n_algorithms, char algorithms[][100], char machine[][4], unsigned long threads[], bool used_shared[], bool correctness[], double elapsed_time[])
{
    char correct[4];
    char shared[4];

    // Print the table headers
    printf("%-46s %-8s %-10s %-13s %-7s %-15s\n", "Algorithm", "Machine", "N. Threads", "Shared Memory", "Correct", "Elapsed Time");

    // Print a line separator after the headers
    printf("---------------------------------------------- -------  ---------- ------------- ------- ------------\n");

    // Print each row of the table
    for (int i = 0; i < n_algorithms; i++)
    {
        if (correctness[i])
            strcpy(correct, "YES");
        else
            strcpy(correct, "NO");

        if (used_shared[i])
            strcpy(shared, "YES");
        else
            strcpy(shared, "NO");

        printf("%-46s %-8s %-10lu %-13s %-7s %-15lf\n", algorithms[i], machine[i], threads[i], shared, correct, elapsed_time[i]);
    }
    printf("\n\n");
}

void write_statistics_csv(int n, char algorithms[][100], double elapsed_times[])
{
    FILE *file = fopen("statistics.csv", "a");
    if (file == NULL)
    {
        printf("Failed to open the file.\n");
        return;
    }

    if (ftell(file) == 0)  // Check if the file is empty
    {
        // Write the header row with algorithm names
        fprintf(file, "N,");
        for (int i = 0; i < 4; i++)
        {
            fprintf(file, "%s", algorithms[i]);
            // Add a comma for all columns except the last one
            if (i < 3)
                fprintf(file, ",");
        }
        fprintf(file, "\n"); // Move to the next row
    }

    fprintf(file, "%d,", n);

    // Write the elapsed time for each algorithm
    for (int i = 0; i < 4; i++)
    {
        fprintf(file, "%lf", elapsed_times[i]);

        // Add a comma for all columns except the last one
        if (i < 3)
            fprintf(file, ",");
    }
    fprintf(file, "\n"); // Move to the next row

    fclose(file);

    printf("CSV file written successfully.\n\n");
}