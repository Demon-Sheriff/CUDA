#include <cuda_runtime.h>
using namespace std;

#define TILE_SIZE 32
#define MASK_WIDTH 5
#define min(x, y) (((x) < (y)) ? (x) : (y))


// store the convolution filter into constant memory    
__constant__ float d_mask[MASK_WIDTH];
// tiled 1d convolution using shared memory
// tile sizes are always even
// convolution filters are always of odd length (also min mask_width = 3)
__global__ void conv_1d_tiled(float *input, float *output, int input_size, int output_size) {

    int tx = threadIdx.x, bx = blockIdx.x;
    int global_idx = tx + bx * TILE_SIZE;
    int r = MASK_WIDTH / 2;

    // declare shared memory storage.
    int w = TILE_SIZE + MASK_WIDTH - 1;
    __shared__ float SM[w];

    // load the core element into shared memory
    if (global_idx < input_size)
        SM[tx + r] = input[global_idx];
    else
        SM[tx + r] = 0.0f; // pad with 0 if we are out of bounds

    // load the left halo
    if (tx - r < 0) {
        SM[tx] = (global_idx - r >= 0) ? input[global_idx - r] : 0.0f;
    }

    // load the right halo
    if (tx + r > TILE_SIZE - 1) {
        SM[tx + 2*r] = (global_idx + r < input_size) ? input[global_idx + r] : 0.0f; 
    }

    // barrier sync -> wait for all the threads to load the elements into SMEM 
    __syncthreads();

    // calculate the convolution.
    float conv_sum = 0.0f;
    for (int s=0; s<MASK_WIDTH; ++s) 
        conv_sum += SM[tx + s] * d_mask[s];
    
    if (global_idx < output_size)
        output[global_idx] = conv_sum;
}

int main() {



    return 0;
}