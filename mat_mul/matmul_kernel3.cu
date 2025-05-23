#include <cuda_runtime.h>
using namespace std;

#define CIEL_DIV(M, N) ((M + (N - 1)) / N)
#define BLOCKSIZE 32 // default access block size for a warp i.e 32 threads in a block

__global__ void kernel3(int M, int N, int K, float *A, float *B, float *C) {

    int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    // this time we iterate over each block and transfer all of it's GMEM to SMEM
    // then compute the result over SMEM
    // keep accumulating the result for each block
    // finally assign the accumulated answer to C[row][col];
    for (int bIdx = 0; bIdx < K; bIdx++) {

        
    }
}
int main() {

    return 0;
}