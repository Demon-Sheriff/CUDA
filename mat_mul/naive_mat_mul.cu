#include <cuda_runtime.h>
#include <stdio.h>

// Naive matrix multiplication kernel with memory access logging
__global__ void matrix_multiply_naive(float *A, float *B, float *C, int m, int n, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C

    if (i < m && j < n)
    {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
        {
            // Log thread IDs, indices, and memory addresses
            printf("Thread (blockIdx.x=%d, blockIdx.y=%d, threadIdx.x=%d, threadIdx.y=%d) "
                   "computing C[%d][%d]: accessing A[%d][%d] at %p, B[%d][%d] at %p\n",
                   blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                   i, j, i, l, (void *)&A[i * k + l], l, j, (void *)&B[l * n + j]);

            sum += A[i * k + l] * B[l * n + j];
        }
        C[i * n + j] = sum;
    }
}

int main()
{
    // Matrix dimensions: A (m x k), B (k x n), C (m x n)
    const int m = 3; // Rows of A, C
    const int k = 4; // Columns of A, rows of B
    const int n = 5; // Columns of B, C

    // Host data
    float h_A[] = {
        1.0, 2.0, 3.0, 4.0,   // A[0][:]
        5.0, 6.0, 7.0, 8.0,   // A[1][:]
        9.0, 10.0, 11.0, 12.0 // A[2][:]
    };
    float h_B[] = {
        1.0, 2.0, 3.0, 4.0, 5.0,      // B[0][:]
        6.0, 7.0, 8.0, 9.0, 10.0,     // B[1][:]
        11.0, 12.0, 13.0, 14.0, 15.0, // B[2][:]
        16.0, 17.0, 18.0, 19.0, 20.0  // B[3][:]
    };
    float h_C[m * n];

    // Device data
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Grid and block configuration
    dim3 threadsPerBlock(4, 4, 1); // 4x4 threads per block
    dim3 blocksPerGrid(
        (n + threadsPerBlock.x - 1) / threadsPerBlock.x, // ceil(n/4) = ceil(5/4) = 2
        (m + threadsPerBlock.y - 1) / threadsPerBlock.y, // ceil(m/4) = ceil(3/4) = 1
        1);                                              // blocksPerGrid = (2, 1, 1)

    // Launch kernel
    matrix_multiply_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy and print results
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nMatrix C (%d x %d):\n", m, n);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f ", h_C[i * n + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}