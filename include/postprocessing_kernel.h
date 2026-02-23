#ifndef POSTPROCESSING_KERNEL_H
#define POSTPROCESSING_KERNEL_H

#include <cuda_runtime.h>
#include <vector>

struct Detection {
    float x,y,w,h;
    float confidence;
    int class_id;
    char class_name[64];
};

__global__ void decodeAndFilterKernel(float* predictions, Detection* output, int* outputCount, int numPredictions, float confThreshold, int imgWidth, int imgHeight, float scale, int offsetX, int offsetY);

__host__ void decodeAndFilterPredictions(float* d_predictions, std::vector<Detection>& detections, float confThreshold, int imgWidth, int imgHeight, int numPredictions, int predictionSize, int inputWidth, int inputHeight);

#endif