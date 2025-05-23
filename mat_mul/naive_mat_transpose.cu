#include <cuda_runtime.h>
using namespace std;


__global__ void naive_transpose(int *input, int *output, int N) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;
    int index = idxY * N + idxY;
    int tp_index = idxX * N + idxY;

    output[index] = input[tp_index];
}
int main() {


    return 0;
}
