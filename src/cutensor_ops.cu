#include "../include/cutensor_ops.h"
#include "../include/error_helpers.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernels for 4D tensor transpose (replaces deprecated cuTENSOR 1.x API)
__global__ void transposeNCHWtoNHWCKernel(const float* __restrict__ input, float* __restrict__ output,
                                          int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int n = idx / (C * H * W);
    int c = (idx / (H * W)) % C;
    int h = (idx / W) % H;
    int w = idx % W;
    int outIdx = ((n * H + h) * W + w) * C + c;
    output[outIdx] = input[idx];
}

__global__ void transposeNHWCtoNCHWKernel(const float* __restrict__ input, float* __restrict__ output,
                                          int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    if (idx >= total) return;
    int n = idx / (H * W * C);
    int h = (idx / (W * C)) % H;
    int w = (idx / C) % W;
    int c = idx % C;
    int outIdx = ((n * C + c) * H + h) * W + w;
    output[outIdx] = input[idx];
}

CutensorOps::CutensorOps() {
    cutensorStatus_t status = cutensorCreate(&handle);
    checkCutensorError(status, "Cutensor handle creation failed");
    printf("Cutensor handle created successfully\n");
}

CutensorOps::~CutensorOps() {
    cutensorStatus_t status = cutensorDestroy(handle);
    checkCutensorError(status, "Cutensor handle destruction failed");
    printf("Cutensor handle destroyed successfully\n");
}

void CutensorOps::transposeTensor4D_NCHW_to_NHWC(float* input, float* output, int N, int C, int H, int W) {
    int total = N * C * H * W;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;
    transposeNCHWtoNHWCKernel<<<numBlocks, blockSize>>>(input, output, N, C, H, W);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "transposeNCHWtoNHWC kernel failed: %s\n", cudaGetErrorString(err));
    }
}

void CutensorOps::transposeTensor4D_NHWC_to_NCHW(float* input, float* output, int N, int C, int H, int W) {
    int total = N * H * W * C;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;
    transposeNHWCtoNCHWKernel<<<numBlocks, blockSize>>>(input, output, N, C, H, W);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "transposeNHWCtoNCHW kernel failed: %s\n", cudaGetErrorString(err));
    }
}

void CutensorOps::permuteTensor(float* input, float* output, int numDims, int64_t* extent, int* permutation) {
    if (numDims == 4 && permutation[0] == 0 && permutation[1] == 2 && permutation[2] == 3 && permutation[3] == 1) {
        transposeTensor4D_NCHW_to_NHWC(input, output, extent[0], extent[1], extent[2], extent[3]);
    } else if (numDims == 4 && permutation[0] == 0 && permutation[1] == 3 && permutation[2] == 1 && permutation[3] == 2) {
        transposeTensor4D_NHWC_to_NCHW(input, output, extent[0], extent[1], extent[2], extent[3]);
    } else {
        fprintf(stderr, "permuteTensor: only NCHW<->NHWC (4D) supported for arbitrary permutation\n");
    }
}

__host__ void demonstrateCutensorTranspose(cutensorHandle_t /*handle*/) {
    printf("Demonstrating 4D Tensor Transpose (NCHW -> NHWC)\n");
    int N = 2, C = 3, H = 4, W = 5;
    int size = N*C*H*W;

    float* h_input = new float[size];
    float* h_output = new float[size];

    for(int n = 0; n < N; n++) {
        for(int c = 0; c < C; c++) {
            for(int h = 0; h < H; h++) {
                for(int w = 0; w < W; w++) {
                    int idx = ((n*C + c)*H + h)*W + w;
                    h_input[idx] = c * 100 + h * 10 + w;
                }
            }
        }
    }
    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, size*sizeof(float));
    cudaMalloc((void**)&d_output, size*sizeof(float));
    cudaMemcpy(d_input, h_input, size*sizeof(float), cudaMemcpyHostToDevice);

    CutensorOps ops;
    ops.transposeTensor4D_NCHW_to_NHWC(d_input, d_output, N, C, H, W);

    cudaMemcpy(h_output, d_output, size*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: \n");
    for(int i = 0;i < size;i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    printf("Cutensor transpose demonstration completed successfully\n");

}