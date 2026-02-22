#ifndef NMS_KERNEL_H
#define NMS_KERNEL_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "postprocessing_kernel.h"

__device__ float calculateIoU(Detection* a, Detection* b);

__global__ void nmsKernel(Detection* detections, bool* keep,int numDetections, float iouThreshold);

__host__ void nonMaximumSuppression(std::vector<Detection>& input, std::vector<Detection>& output, float iouThreshold);

#endif