#ifndef ERROR_HELPERS_H
#define ERROR_HELPERS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutensor.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA Error Checking

inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// CUBLAS Error Checking

inline void checkCublasError(cublasStatus_t err, const char* msg) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS Error: %s: %s\n", msg, msg);
        exit(EXIT_FAILURE);
    }
}

// CUTENSOR Error Checking

inline void checkCutensorError(cutensorStatus_t err, const char* msg) {
    if (err != CUTENSOR_STATUS_SUCCESS) {
        fprintf(stderr, "CUTENSOR Error: %s: %s\n", msg, msg);
        exit(EXIT_FAILURE);
    }
}

#define TRY_TORCH(expr) \
try {expr;} \
catch(const c10: Error& e) { \
    fprintf(stderr, "Torch error: %s\n", e.what()); \
    exit(EXIT_FAILURE); \
}

#endif
