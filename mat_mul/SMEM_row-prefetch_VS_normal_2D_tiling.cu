#include <iostream>
#include <vector>
#include <cstdlib> // For rand(), srand()
#include <ctime>   // For time()
#include <iomanip> // For std::fixed, std::setprecision

#include <cuda_runtime.h>

// Define TILE_SIZE, must be consistent with kernel assumptions
#define TILE_SIZE 32
#define min(x, y) ((x) < (y) ? (x) : (y))

// CUDA Error Checking Macro
#define CUDA_CHECK(err)                                                       \
    {                                                                         \
        cudaError_t err_ = (err);                                             \
        if (err_ != cudaSuccess)                                              \
        {                                                                     \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ \
                      << ": " << cudaGetErrorString(err_) << std::endl;       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// Kernel 1: Row Prefetching Tiling (User's version)
__global__ void rp_tiling(const float *A, const float *B, float *C, int M, int N, int K) {
    // blockDim.x = blockDim.y = TILE_SIZE
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = ty + by * TILE_SIZE, col = tx + bx * TILE_SIZE;

    // Declare shared memory storage
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    // Iterate over the tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < numTiles; ++tile)
    {

        // Populate tileA using row-major prefetching strategy by tx == 0 threads
        if (tx == 0)
        {
            // User's original 'lim'. This means tileA is logically TILE_SIZE rows by 'lim' columns.
            // Elements tileA[ty][t] for t >= lim are not initialized by this block.
            int lim = min(TILE_SIZE, N - tile * TILE_SIZE); // This was N - tile in user's code
            for (int t = 0; t < lim; ++t)
            {
                // Boundary check for A: rows (row < M), columns ((t + tile*TILE_SIZE) < N)
                tileA[ty][t] = (row < M && (t + tile * TILE_SIZE) < N) ? A[row * N + t + tile * TILE_SIZE] : 0.0f;
            }
        }

        // Populate tileB: each thread loads one element
        // Boundary check for B: rows ((tile*TILE_SIZE + ty) < N), columns (col < K)
        tileB[ty][tx] = ((tile * TILE_SIZE + ty) < N && col < K) ? B[(tile * TILE_SIZE + ty) * K + col] : 0.0f;

        // Barrier synchronization
        __syncthreads();

        // Accumulate the dot product for the current tile
        // User's original 'limit'. Computation loop will only access initialized parts of tileA
        // if limit is the same as lim used for loading tileA's columns.
        int limit = min(TILE_SIZE, N - tile * TILE_SIZE); // This was N - tile in user's code
        for (int x = 0; x < limit; x++)
        {
            sum += tileA[ty][x] * tileB[x][tx];
        }

        // Barrier synchronization
        __syncthreads();
    }
    // Assign the dot product to the output matrix (C)
    if (row < M && col < K)
        C[row * K + col] = sum;
}

// Kernel 2: Normal Tiling (Each thread loads one element for A and B)
__global__ void normal_tiling(const float *A, const float *B, float *C, int M, int N, int K)
{
    // blockDim.x = blockDim.y = TILE_SIZE
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = ty + by * TILE_SIZE, col = tx + bx * TILE_SIZE;

    // Declare shared memory storage
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    // Iterate over the tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < numTiles; ++tile)
    {

        // Populate tileA: each thread loads one element
        // Boundary check for A: rows (row < M), columns ((tile*TILE_SIZE + tx) < N)
        tileA[ty][tx] = (row < M && (tile * TILE_SIZE + tx) < N) ? A[row * N + tile * TILE_SIZE + tx] : 0.0f;

        // Populate tileB: each thread loads one element
        // Boundary check for B: rows ((tile*TILE_SIZE + ty) < N), columns (col < K)
        tileB[ty][tx] = ((tile * TILE_SIZE + ty) < N && col < K) ? B[(tile * TILE_SIZE + ty) * K + col] : 0.0f;

        // Barrier synchronization
        __syncthreads();

        // Accumulate the dot product for the current tile
        // User's original 'limit'. If tileA/tileB are zero-padded, this is correct.
        // A more optimized limit would be min(TILE_SIZE, N - tile * TILE_SIZE)
        int limit = min(TILE_SIZE, N - tile * TILE_SIZE); // This was N - tile in user's code
        for (int x = 0; x < limit; x++)
        {
            sum += tileA[ty][x] * tileB[x][tx];
        }

        // Barrier synchronization
        __syncthreads();
    }

    // Assign the dot product to the output matrix (C)
    if (row < M && col < K)
        C[row * K + col] = sum;
}

// Host function to initialize matrix with random values
void init_matrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        mat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 10.0f); // Values between 0 and 10
    }
}

// Structure to hold matrix dimensions
struct MatrixSize
{
    int M, N, K;
};

int main()
{
    srand(static_cast<unsigned int>(time(0))); // Seed random number generator

    // Define matrix sizes to test
    std::vector<MatrixSize> sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {512, 1024, 256},
        {1023, 767, 1279} // Non-multiple of TILE_SIZE
    };

    int warmup_runs = 10;
    int benchmark_iterations = 30;

    std::cout << std::fixed << std::setprecision(3);
    int x = 1;
    for (const auto &size : sizes)
    {
        int M = size.M;
        int N = size.N;
        int K = size.K;

        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Matrix Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
        std::cout << "TILE_SIZE: " << TILE_SIZE << std::endl;

        // Allocate host memory
        float *h_A = new float[M * N];
        float *h_B = new float[N * K];
        float *h_C_rp = new float[M * K];     // Result from rp_tiling
        float *h_C_normal = new float[M * K]; // Result from normal_tiling

        // Initialize host matrices
        init_matrix(h_A, M, N);
        init_matrix(h_B, N, K);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));

        // CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Define grid and block dimensions
        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        dim3 dimGrid((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        // Warm-up runs
        if (x == 1) { // warm up only on the first iteration.
            std::cout << "Performing Warm-up runs..." << std::endl;
            for (int i = 0; i < warmup_runs; ++i)
            {
                rp_tiling<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
                CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel completion
                normal_tiling<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            --x;
        }

        // --- Benchmark rp_tiling kernel ---
        std::cout << "Benchmarking Row Prefetching Tiling (rp_tiling)..." << std::endl;
        // Timed benchmark runs
        float total_time_rp = 0.0f;
        for (int i = 0; i < benchmark_iterations; ++i)
        {
            CUDA_CHECK(cudaMemset(d_C, 0, M * K * sizeof(float))); // Clear output buffer
            CUDA_CHECK(cudaEventRecord(start));
            rp_tiling<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for the event to complete
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_time_rp += ms;
        }
        CUDA_CHECK(cudaMemcpy(h_C_rp, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "  Avg. execution time (rp_tiling): " << (total_time_rp / benchmark_iterations) << " ms" << std::endl;

        // --- Benchmark normal_tiling kernel ---
        std::cout << "Benchmarking Normal Tiling (normal_tiling)..." << std::endl;

        // Timed benchmark runs
        float total_time_normal = 0.0f;
        for (int i = 0; i < benchmark_iterations; ++i)
        {
            CUDA_CHECK(cudaMemset(d_C, 0, M * K * sizeof(float))); // Clear output buffer
            CUDA_CHECK(cudaEventRecord(start));
            normal_tiling<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for the event to complete
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_time_normal += ms;
        }
        CUDA_CHECK(cudaMemcpy(h_C_normal, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "  Avg. execution time (normal_tiling): " << (total_time_normal / benchmark_iterations) << " ms" << std::endl;

        // Clean up
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        delete[] h_A;
        delete[] h_B;
        delete[] h_C_rp;
        delete[] h_C_normal;

        // Optional: Add verification logic here to compare h_C_rp and h_C_normal
        // For example, check if elements are close within a tolerance.
    }
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Benchmarking complete." << std::endl;

    return 0;
}