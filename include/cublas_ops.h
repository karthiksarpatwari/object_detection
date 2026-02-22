#ifndef CUBLAS_OPS_H
#define CUBLAS_OPS_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "error_helpers.h"

class CublasOps {
    public:
        CublasOps();
        ~CublasOps();
        void matrixMultiply(float* A, float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta = 0.0f);
        void vectorMatrixMultiply(float* vector, float* matrix, float* result, int vectorSize, int matrixCols);
        void batchMatrixMultiply(float** A, float** B, float** C, int M, int N, int K, int batchSize);
        cublasHandle_t getHandle() const { return handle;}

    private:
        cublasHandle_t handle;

};

__host__ void demonstrateCublasMatmul(cublasHandle_t handle);

#endif
