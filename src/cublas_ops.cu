#include "../include/cublas_ops.h"
#include "../include/error_helpers.h"
#include <stdio.h>

CublasOps::CublasOps() {
    cublasStatus_t status = cublasCreate(&handle);
    checkCublasError(status, "Cublas handle creation failed");
    printf("Cublas handle created successfully\n");
}

CublasOps::~CublasOps() {
    cublasStatus_t status = cublasDestroy(handle);
    checkCublasError(status, "Cublas handle destruction failed");
    printf("Cublas handle destroyed successfully\n");
}

void CublasOps::matrixMultiply(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    //  Matrix A (MxK), Matrix B (KxN), Matrix C (MxN)
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, N, &beta, C, M);
    checkCublasError(status, "Cublas matrix multiplication failed");
    printf("Cublas matrix multiplication completed successfully\n");
}

void CublasOps::vectorMatrixMultiply(float* vector, float* matrix, float* result, int vectorSize, int matrixCols) {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasSgemv(handle, CUBLAS_OP_N, vectorSize, matrixCols, &alpha, matrix, vectorSize, vector, 1, &beta, result, 1);
    checkCublasError(status, "Cublas vector matrix multiplication failed");
    printf("Cublas vector matrix multiplication completed successfully\n");
}

void CublasOps::batchMatrixMultiply(float** A, float** B, float** C, int M, int N, int K, int batchCount) {

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,(const float**)A, M, (const float**)B, N, &beta, C, M, batchCount);
    checkCublasError(status, "Cublas batch matrix multiplication failed");
    printf("Cublas batch matrix multiplication completed successfully\n");
}

__host__ void demonstrateCublasMatmul(cublasHandle_t handle) {

    printf("Demonstrating Cublas Matrix Multiplication\n");
    int M = 2, N = 3, K = 4;

    float h_A[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_B[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float h_C[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;

    cudaMalloc((void**)&d_A, 16 * sizeof(float));
    cudaMalloc((void**)&d_B, 12 * sizeof(float));
    cudaMalloc((void**)&d_C, 8 * sizeof(float));    

    cudaMemcpy(d_A, h_A, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 12 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, 8 * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, N, &beta, d_C, M);
    checkCublasError(status, "Cublas matrix multiplication failed");
    printf("Cublas matrix multiplication completed successfully\n");

    cudaMemcpy(h_C, d_C, 8 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result: \n");
    for (int i = 0; i < 8; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("Cublas matrix multiplication demonstration completed successfully\n");
}