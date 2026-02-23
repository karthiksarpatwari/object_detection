#include "../include/cutensor_ops.h"
#include "../include/error_helpers.h"
#include <stdio.h>

CutensorOps::CutensorOps() {
    cutensorStatus_t status = cutensorCreate(&handle);
    checkCutensorError(status, "Cutensor handle creation failed");
    printf("Cutensor handle created successfully\n");
}

CutensorOps::~CutensorOps() {
    cutensorStatus_t status = cutensorDestroy(handle);
    checkCutensorError(status, "Cutensor handle destruction failed");
    printf("Cutensor handle destroyed successfully\n");}



void CutensorOps::transposeTensor4D_NCHW_NHCW(float* d_input, float* d_output, int height, int width, int channels) {
    cutensorTensorDescriptor_t descInput, descOutput;
    cutensorStatus_t status;

    int64_t extentInput[4] = {N, C, H, W};
    int64_t strideInput[4] = {C*H*W, H*W, W, 1};

    status = cutensorInitTensorDescriptor(&handle, &descInput, 4, extentInput, strideInput, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY);
    checkCutensorError(status, "Cutensor init tensor descriptor failed");

    int64_t extentOutput[4] = {N, C, H, W};
    int64_t strideOutput[4] = {C*H*W, H*W, W, 1};

    status = cutensorInitTensorDescriptor(&handle, &descOutput, 4, extentOutput, strideOutput, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY);
    checkCutensorError(status, "Cutensor init tensor descriptor failed");
    int64_t extentOutput[4] = {N, H, W, C};
    int64_t strideOutput[4] = {C*H*W, C*W,C, 1};

    status = cutensorInitTensorDescriptor(&handle, &descOutput, 4, extentOutput, strideOutput, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY);
    checkCutensorError(status, "Cutensor init tensor descriptor failed");

    int32_t mode[4] = {0,2,3,1};
    float alpha = 1.0f;

    status = cutensorPermutation(&handle, &alpha,input,&descInput,mode,output,&descOutput,CUTENSOR_R_32F,0);
    checkCutensorError(status, "Cutensor permutation failed");
    printf("Cutensor permutation completed successfully\n");
}
void CutensorOps::transposeTensor4D_NHWC_NCHW(float* d_input, float* d_output, int height, int width, int channels) {
    
    cutensorTensorDescriptor_t descInput, descOutput;
    cutensorStatus_t status;

    int64_t extentInput[4] = {N, C, W, C};
    int64_t strideInput[4] = {C*H*W, C*W, C, 1};

    status = cutensorInitTensorDescriptor(&handle, &descInput, 4, extentInput, strideInput, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY);
    checkCutensorError(status, "Cutensor init tensor descriptor failed");

    int64_t extentOutput[4] = {N, C, H, W};
    int64_t strideOutput[4] = {C*H*W, H*W, W, 1};

    status = cutensorInitTensorDescriptor(&handle, &descOutput, 4, extentOutput, strideOutput, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY);
    checkCutensorError(status, "Cutensor init tensor descriptor failed");

    int32_t mode[4] = {0,3,1,2};
    float alpha = 1.0f;

    status = cutensorPermutation(&handle, &alpha,input,&descInput,mode,output,&descOutput,CUTENSOR_R_32F,0);
    checkCutensorError(status, "Cutensor permutation failed");
    printf("Cutensor permutation completed successfully\n");
}

__host__ void demonstrateCutensorTranspose(cutensorHandle_t handle) {
    printf("Demonstrating Cutensor Transpose\n");
    int N = 2, C = 3, H = 4, W = 5;
    int size = N*C*H*W;

    float* h_input = new float[size]; //(float*)malloc(sizesizeof(float));
    float* h_output = new float[size]; //(float*)malloc(N*C*H*W*sizeof(float));

    for(int n = 0;n < N;n++) {
        for(int c = 0;c < C;c++) {
            for(int h = 0;h < H;h++) {
                for(int w = 0;w < W;w++) {
                    int idx = ((n*C + c)*H + h)*W + w;
                    h_input[idx] =c *100 + h * 10 + w;
                }
            }
        }
    }
    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, size*sizeof(float));
    cudaMalloc((void**)&d_output, size*sizeof(float));
    cudaMemcpy(d_input, h_input, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size*sizeof(float), cudaMemcpyHostToDevice);

    cutensorTensorDescriptor_t descInput, descOutput;
    int64_t extentInput[4] = {N, C, H, W};
    int64_t strideInput[4] = {C*H*W, H*W, W, 1};
    int64_t extentOutput[4] = {N, H, W, C};
    int64_t strideOutput[4] = {C*H*W, C*W,C, 1};

    status = cutensorInitTensorDescriptor(&handle, &descInput, 4, extentInput, strideInput, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY);
    checkCutensorError(status, "Cutensor init tensor descriptor failed");

    status = cutensorInitTensorDescriptor(&handle, &descOutput, 4, extentOutput, strideOutput, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY);
    checkCutensorError(status, "Cutensor init tensor descriptor failed");

    float alpha =1.0f;
    int32_t mode[4] = {0,2,3,1};
    status = cutensorPermutation(&handle, &alpha,d_input,&descInput,mode,d_output,&descOutput,CUTENSOR_R_32F,0);
    checkCutensorError(status, "Cutensor permutation failed");
    printf("Cutensor permutation completed successfully\n");

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