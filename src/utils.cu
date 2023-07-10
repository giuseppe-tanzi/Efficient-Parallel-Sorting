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