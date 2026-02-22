#ifndef CUTENSOR_OPS_H
#define CUTENSOR_OPS_H

#include <cutensor.h>
#include <cuda_runtime.h>
#include "error_helpers.h"

class CutensorOps {
    public:
        CutensorOps();
        ~CutensorOps();
 
        void transposeTensor4D_NCHW_to_NHWC(float* input, float* output, int N, int C, int H, int W);
        void transposeTensor4D_NHWC_to_NCHW(float* input, float* output, int N, int C, int H, int W);
        void permuteTensor(float* input, float* output, int numDims, int64_t* extent, int* permutation);
        cutensorHandle_t getHandle() const { return handle;}

    private:
        cutensorHandle_t handle;
};

__host__ void demonstrateCutensorTranspose(cutensorHandle_t handle);

#endif  