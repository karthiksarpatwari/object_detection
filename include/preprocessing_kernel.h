#ifndef PREPROCESSING_KERNEL_H
#define PREPROCESSING_KERNEL_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "error_helpers.h"
#include "image_loader.h"

struct PreprocessParams {
    float mean[3];
    float std[3];
    float padValue;

};
// Default preprocessing parameters for ImageNet
const PreprocessParams DEFAULT_PARAMS = {
    {0.485f, 0.456f, 0.406f},
    {0.229f, 0.224f, 0.225f},
    0.5f
};

__global__ void letterboxResizeKernel(unsigned char* src, float* dst, int srcW, int srcH, int dstW, int dstH, int channels, float padValue, float scale, int offsetX, int offsetY);

__global__ void normalizeAndTransposeKernel(unsigned char* src, float* dst, int width, int height, int channels, float* mean, float* std);

__host__ void preprocessImage(Image* image, float* d_output, int targetW, int targetH, PreprocessParams params = DEFAULT_PARAMS);

__host__ void preProcessImageBatch(std::vector<Image*>& images, float* d_output, int targetW, int targetH, PreprocessParams params = DEFAULT_PARAMS);

#endif

