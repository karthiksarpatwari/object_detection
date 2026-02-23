#include "../include/nms_kernel.h"
#include "../include/error_helpers.h"
#include <stdio.h>
#include <algorithm>

__device__ float calculateIoU(Detection* a, Detection* b) {
    float a_x1 = a->x - a->w/2.0f;
    float a_y1 = a->y - a->h/2.0f;
    float a_x2 = a->x + a->w/2.0f;
    float a_y2 = a->y + a->h/2.0f;

    float b_x1 = b->x - b->w/2.0f;
    float b_y1 = b->y - b->h/2.0f;
    float b_x2 = b->x + b->w/2.0f;
    float b_y2 = b->y + b->h/2.0f;

    float inter_x1 = fmaxf(a_x1, b_x1);
    float inter_y1 = fmaxf(a_y1, b_y1);

    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);

    float inter_area = inter_w * inter_h;
    float a_area = a->w * a->h;
    float b_area = b->w * b->h;

    if (union_area < 1e-6f) return 0.0f;

    return inter_area / union_area;

}

__global__ void nmsKernel(Detection* detections, bool* keep, int numDetections, float iouThreshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numDetections || !keep[idx]) return;
    Detection* current = &detections[idx];
    
    for (int i = idx + 1; i < numDetections; i++) {
        Detection* other = &detections[i];
        float iou = calculateIoU(current, other);
        if (iou > iouThreshold) {
            keep[i] = false;
        }
    }
}

__host__ void nonMaximumSuppression(std::vector<Detection>& input, std::vector<Detection>& output, float iouThreshold) {

    if (input.empty()) return;

    std::sort(input.begin(), input.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    Detection* d_detections;
    bool* d_keep;

    cudaMalloc(&d_detections, input.size() * sizeof(Detection));
    cudaMalloc(&d_keep, input.size() * sizeof(bool));
    cudaMemcpy(d_detections, input.data(), input.size() * sizeof(Detection), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keep, keep.data(), input.size() * sizeof(bool), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (input.size() + blockSize - 1) / blockSize;
    nmsKernel<<<numBlocks, blockSize>>>(d_detections, d_keep, input.size(), iouThreshold);
    checkCudaError(cudaGetLastError(), "NMS kernel execution failed");
    cudaDeviceSynchronize();

    cudaMemcpy(h_keep, d_keep, input.size() * sizeof(bool), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < input.size(); i++) {
        if (h_keep[i]) {
            output.push_back(input[i]);
        }
    }

    delete[] h_keep;
    cudaFree(d_detections);
    cudaFree(d_keep);
}