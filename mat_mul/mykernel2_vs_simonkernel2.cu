#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h> // For fabs in correctness check

#define BLOCKSIZE 16

__global__ void matrix_multiply_naive(float *A, float *B, float *C, int m, int n, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n)
    {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
        {
            sum += A[i * k + l] * B[l * n + j];
        }
        C[i * n + j] = sum;
    }
}

__global__ void matmul_tp(float *A, float *B_t, float *C, int m, int n, int k)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
        {
            sum += A[row * k + l] * B_t[col * k + l];
        }
        C[row * n + col] = sum;
    }
}

__global__ void transpose(float *B, float *BT, int k, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && l < k)
    {
        BT[j * k + l] = B[l * n + j];
    }
}

__global__ void matrix_multiply_naive_2(float *A, float *B, float *C, int m, int n, int k)
{
    int j = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    int i = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
       
    if (i < m && j < n)
    {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
        {
            sum += A[i * k + l] * B[l * n + j];
        }
        C[i * n + j] = sum;
    }
}

void matrix_multiply_cpu(float *A, float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++){
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

bool check_correctness(float *cpu_C, float *gpu_C, int m, int n, float epsilon = 1e-4)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int idx = i * n + j;
            if (fabs(cpu_C[idx] - gpu_C[idx]) > epsilon)
            {
                printf("Mismatch at C[%d][%d]: CPU = %.6f, GPU = %.6f\n", i, j, cpu_C[idx], gpu_C[idx]);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    // Adjusted matrix dimensions
    const int m = 640;  // Rows of A, C
    const int k = 3200; // Columns of A, rows of B
    const int n = 6400;  // Columns of B, C

    // Host data
    printf("Allocating host memory\n");
    float *h_A = (float *)malloc(m * k * sizeof(float));
    float *h_B = (float *)malloc(k * n * sizeof(float));
    float *h_C = (float *)malloc(m * n * sizeof(float));
    float *h_C_ref = (float *)malloc(m * n * sizeof(float)); // For CPU result
    float *h_C_gpu = (float *)malloc(m * n * sizeof(float)); // To copy GPU result
    float *h_BT = (float *)malloc(n * k * sizeof(float));    // For transposed B

    // Initialize with random data
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < m * k; i++)
        h_A[i] = (float)(rand()) / (float)RAND_MAX;
    for (int i = 0; i < k * n; i++)
        h_B[i] = (float)(rand()) / (float)RAND_MAX;
    for (int i = 0; i < m * n; i++)
        h_C[i] = 0.0f;

    // Compute reference result on CPU
    printf("Computing CPU output for mat_mul\n");
    matrix_multiply_cpu(h_A, h_B, h_C_ref, m, n, k);

    printf("Allocating device memory\n");
    // Device data
    float *d_A, *d_B, *d_BT, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_BT, n * k * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Grid and block configuration
    dim3 threadsPerBlock(32, 1, 1);
    dim3 blocksPerGrid_naive((n + threadsPerBlock.x - 1) / threadsPerBlock.x, m, 1);
    dim3 blocksPerGrid_tp((n + threadsPerBlock.x - 1) / threadsPerBlock.x, m, 1);
    dim3 blocksPerGrid_trans((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (k + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
    dim3 blocksPerGrid_simon((n + BLOCKSIZE - 1) / BLOCKSIZE, (m + BLOCKSIZE - 1) / BLOCKSIZE, 1);

    // Warm-up runs (4 iterations)
    printf("Performing warm up runs\n");
    for (int warm = 0; warm < 4; warm++)
    {
        cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice); // Reset C
        matrix_multiply_naive<<<blocksPerGrid_naive, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
        cudaDeviceSynchronize();

        cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice); // Reset C
        transpose<<<blocksPerGrid_trans, threadsPerBlock>>>(d_B, d_BT, k, n);
        matmul_tp<<<blocksPerGrid_tp, threadsPerBlock>>>(d_A, d_BT, d_C, m, n, k);
        cudaDeviceSynchronize();

        cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice); // Reset C
        matrix_multiply_naive_2<<<blocksPerGrid_simon, dim3(BLOCKSIZE * BLOCKSIZE, 1, 1)>>>(d_A, d_B, d_C, m, n, k);
        cudaDeviceSynchronize();
    }

    // Correctness Test
    printf("Testing correctness...\n");

    // Naive Kernel
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice); // Reset C
    matrix_multiply_naive<<<blocksPerGrid_naive, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Naive Kernel Correctness: %s\n", check_correctness(h_C_ref, h_C_gpu, m, n) ? "PASS" : "FAIL");

    // Transposed Kernel
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice); // Reset C
    transpose<<<blocksPerGrid_trans, threadsPerBlock>>>(d_B, d_BT, k, n);
    matmul_tp<<<blocksPerGrid_tp, threadsPerBlock>>>(d_A, d_BT, d_C, m, n, k);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Transposed Kernel Correctness: %s\n", check_correctness(h_C_ref, h_C_gpu, m, n) ? "PASS" : "FAIL");

    // Simon's Kernel
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice); // Reset C
    matrix_multiply_naive_2<<<blocksPerGrid_simon, dim3(BLOCKSIZE * BLOCKSIZE, 1, 1)>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Simon's Kernel Correctness: %s\n", check_correctness(h_C_ref, h_C_gpu, m, n) ? "PASS" : "FAIL");

    // Benchmarking (average over 20 runs)
    printf("Benchmarking (average over 20 runs)...\n");
    float times[3] = {0.0f, 0.0f, 0.0f}; // Accumulate times for averaging
    const int num_runs = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Naive Kernel
    for (int run = 0; run < num_runs; run++)
    {
        cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice); // Reset C
        cudaEventRecord(start);
        matrix_multiply_naive<<<blocksPerGrid_naive, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        times[0] += elapsed;
    }
    times[0] /= num_runs;
    printf("Naive Kernel Average Time: %.2f ms\n", times[0]);

    // Transposed Kernel (Total time: transpose + matmul)
    for (int run = 0; run < num_runs; run++)
    {
        cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice); // Reset C
        cudaEventRecord(start);
        transpose<<<blocksPerGrid_trans, threadsPerBlock>>>(d_B, d_BT, k, n);
        matmul_tp<<<blocksPerGrid_tp, threadsPerBlock>>>(d_A, d_BT, d_C, m, n, k);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        times[1] += elapsed;
    }
    times[1] /= num_runs;
    printf("Transposed Kernel Average Time (Transpose + Matmul): %.2f ms\n", times[1]);

    // Simon's Kernel
    for (int run = 0; run < num_runs; run++)
    {
        cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice); // Reset C
        cudaEventRecord(start);
        matrix_multiply_naive_2<<<blocksPerGrid_simon, dim3(BLOCKSIZE * BLOCKSIZE, 1, 1)>>>(d_A, d_B, d_C, m, n, k);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        times[2] += elapsed;
    }
    times[2] /= num_runs;
    printf("Simon's Kernel Average Time: %.2f ms\n", times[2]);

    // Calculate GFLOPS (approximate)
    double gflops = 2.0 * m * n * k * 1e-9; // 2 operations per multiply-add 
    printf("Naive GFLOPS: %.2f\n", gflops / (times[0] / 1000.0));
    printf("Transposed GFLOPS: %.2f\n", gflops / (times[1] / 1000.0));
    printf("Simon's GFLOPS: %.2f\n", gflops / (times[2] / 1000.0));

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_BT);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    free(h_C_gpu);
    free(h_BT);

    return 0;
}