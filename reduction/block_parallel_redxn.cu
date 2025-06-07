#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;

#define TILE_SIZE 1024

__global__ void block_sum(float *input, float *output, int n) {

    int tx = threadIdx.x, bx = blockIdx.x;
    __shared__ float sm[TILE_SIZE]; // TILE_SIZE threads in the starting block for now.

    sm[tx] = (bx * blockDim.x + tx) < n ? input[bx * blockDim.x + tx] : 0.0f;
    __syncthreads();

    // applying redxn
    for (int stride=TILE_SIZE/2; stride>0; stride >>= 1) {
        if (tx < stride) {
            sm[tx] += sm[tx + stride];
        }
        __syncthreads(); // barrier sync for current level.
    }

    if (tx == 0) 
        output[bx] = sm[0];
}

float cpu_sum(float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += arr[i];
    }
    return sum;
}

int main() {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cout << "Performing Warm-up runs...\n";
    for (int N = 0; N <= 20; N += 1) {
        float *h_input = new float[N];
        float *h_output;
        for (int i = 0; i < N; ++i)
            h_input[i] = 1.0f;

        h_output = new float[(N + TILE_SIZE - 1) / TILE_SIZE];

        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, ((N + TILE_SIZE - 1) / TILE_SIZE) * sizeof(float));

        cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE);

        block_sum<<<grid, block>>>(d_input, d_output, N);
        cudaMemcpy(h_output, d_output, grid.x * sizeof(float), cudaMemcpyDeviceToHost);

        float gpu_result = 0.0f;
        for (int i = 0; i < grid.x; ++i)
            gpu_result += h_output[i];

        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
    }

    cout << "Size\tCPU Time (ms)\tGPU Time (ms)\tResult Check\n";

    for (int N = 0; N <= 1000000; N += 1000) {
        float *h_input = new float[N];
        float *h_output;
        for (int i = 0; i < N; ++i)
            h_input[i] = 1.0f;

        h_output = new float[(N + TILE_SIZE - 1) / TILE_SIZE];

        auto cpu_start = chrono::high_resolution_clock::now();
        float cpu_result = cpu_sum(h_input, N);
        auto cpu_end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> cpu_time = cpu_end - cpu_start;

        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, ((N + TILE_SIZE - 1) / TILE_SIZE) * sizeof(float));

        cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE);

        cudaEventRecord(start);
        block_sum<<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_output, d_output, grid.x * sizeof(float), cudaMemcpyDeviceToHost);

        float gpu_result = 0.0f;
        for (int i = 0; i < grid.x; ++i)
            gpu_result += h_output[i];

        bool correct = fabs(cpu_result - gpu_result) < 1e-3;

        if (N == 10 || N == 100 || N == 1000 || N == 5000 || N == 10000 || N == 100000 || N == 1000000) {

            cout << N << "\t" << cpu_time.count()
                    << "\t\t" << ms
                    << "\t\t" << (correct ? "PASS" : "FAIL") << "\n";
        }

        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}