#include "../include/postprocessing_kernel.h"
#include "../include/error_helpers.h"
#include <stdio.h>

__global__ void decodeAndFilterKernel(float* predictions, Detection* output, int* outputCount, int numPredictions, float confThreshold, int imgWidth, int imgHeight, float scale, int offsetX, int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= numPredictions) return;
    
    float* pred = &predictions[idx*85]; 

    float objConf = pred[4];

    if (objConf < confThreshold) return;

    float maxClassProb = 0.0f;
    int maxClassIdx = 0;

    for (int c = 5; c < 85; c++) {
        if (pred[c] > maxClassProb) {
            maxClassProb = pred[c];
            maxClassIdx = c - 5;
        }
    }

    float conf = objConf * maxClassProb;

    if (conf < confThreshold) return;
    
    float x_center = pred[0];
    float y_center = pred[1];
    float width = pred[2];
    float height = pred[3];

    x_center = x_center - width/2.0f;
    y_center = y_center - height/2.0f;
    width = fmaxf(0.0f, fminf(width, float(imgWidth)));
    height = fmaxf(0.0f, fminf(height, float(imgHeight)));
    
    int outIdx = atomicAdd(outputCount, 1);

    if (outIdx < 10000) {
        output[outIdx].x = x_center;
        output[outIdx].y = y_center;
        output[outIdx].width = width;
        output[outIdx].height = height;
        output[outIdx].class_id= maxClassIdx;
        output[outIdx].confidence = conf;
    }
}

__host__ void decodeAndFilterPredictions(float* d_predictions, std::vector<Detection>& detections, float confThreshold, int imgWidth, int imgHeight, int numPredictions, int predctionSize) {

    float scale = fminf((float)inputWidth/imgWidth, (float)inputHeight/imgHeight);
    int  newW = (int)(imgWidth * scale);
    int newH = (int)(imgHeight * scale);
    int offsetX = (inputWidth - newW) / 2;
    int offsetY = (inputHeight - newH) / 2;

    Detection* d_output;
    int *d_outputCount;
    cudaMalloc(&d_output, 10000 * sizeof(Detection));
    cudaMalloc(&d_outputCount, sizeof(int));
    cudaMemset(d_outputCount, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numPredictions + threadsPerBlock - 1) / threadsPerBlock;

    decodeAndFilterKernel<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_output, d_outputCount, numPredictions, confThreshold, imgWidth, imgHeight, scale, offsetX, offsetY);
    checkCudaErrors(cudaGetLastError(),"Decode and Filter Kernel failed");

    cudaDeviceSynchronize();

    int h_outputCount = 0;
    if (h_outputCount > 0) {
        Detection* h_detections = new Detection[h_outputCount];
        cudaMemcpy(h_detections, d_output, h_outputCount * sizeof(Detection), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < h_outputCount; i++) {
            detections.push_back(h_detections[i]);
        }
        delete[] h_detections;
    }

    cudaFree(d_output);
    cudaFree(d_outputCount);
}