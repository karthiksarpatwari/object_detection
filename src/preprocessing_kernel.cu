#include "../include/preprocessing_kernel.h"
#include "../include/error_helpers.h"
#include <stdio.h>



__global__ void letterboxResizeKernel(unsigned char* src, float* dst, int srcW, int srcH, int dstW, int dstH, int channels, float padValue, float scale, int offsetX, int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

   int newW = (int)(srcW * scale);
   int newH = (int)(srcH*scale);

   bool inImage = x >= offsetX && x < offsetX + newW && y >= offsetY && y < offsetY + newH;

   if(inImage) {

    int srcX = (x - offsetX) * scale;
    int srcY = (y - offsetY) * scale;

    srcX = min(srcX, srcW - 1);
    srcY = min(srcY, srcH - 1);
    
    for (int c = 0; c < channels; c++) {
        int srcIdx = (srcY * srcW + srcX) * channels + c;
        int dstIdx = (c * dstH + y) * dstW + x;
        dst[dstIdx] = src[srcIdx]/255.0f;
    }
   } else {
    for (int c = 0; c < channels; c++) {
        int dstIdx = (c * dstH + y) * dstW + x;
        dst[dstIdx] = padValue;
    }
   }
}

__global__ void normalizeKernel(float* data, int width, int height, int channels, float* mean, float* std) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        int idx = (c*height+y)*width + x; //(y * width + x) * channels + c;
        data[idx] = (data[idx] - mean[c]) / std[c];
    }
}

__host__ void preprocessImage(Image* image, float* d_output, int targetW, int targetH, PreprocessParams params) {
    if(image == NULL || image->data == NULL) { 
        fprintf(stderr, "Error: Failed to preprocess image: %s\n", image->filename);
        return;
    }

   float scale = fminf(targetW / (float)image->width, targetH / (float)image->height);
   int newW = (int)(image->width * scale);
   int newH = (int)(image->height * scale);

   int offsetX = (targetW - newW) / 2;
   int offsetY = (targetH - newH) / 2;

   if(!image->deviceAllocated) {
    copyImageToDevice(image);
   }
   
   float *d_mean;
   float *d_std;

   cudaMalloc(&d_mean, 3 * sizeof(float));
   cudaMalloc(&d_std, 3 * sizeof(float));

   cudaMemcpy(d_mean, params.mean, 3 * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_std, params.std, 3 * sizeof(float), cudaMemcpyHostToDevice);

   dim3 blockSize(16,16);
   dim3 gridSize((targetW + 15)/16, (targetH + 15)/16);

   letterboxResizeKernel<<<gridSize, blockSize>>>(image->data, d_output, image->width, image->height, targetW, targetH, image->channels, params.padValue, scale, offsetX, offsetY);
   checkCudaError(cudaGetLastError(),"Letterbox Kernel failed");

   normalizeKernel<<<gridSize, blockSize>>>(d_output, targetW, targetH, image->channels, d_mean, d_std);
   checkCudaError(cudaGetLastError(),"Normalize Kernel failed");

   cudaDeviceSynchronize();

   // Clean up
   cudaFree(d_mean);
   cudaFree(d_std);
}

// Can be optimized using CUDA streams
__host__ void preProcessImageBatch(std::vector<Image*>& images, float* d_output, int targetW, int targetH, PreprocessParams params) {
    int batchStride = 3*targetW*targetH;
    for(size_t i = 0; i < images.size(); i++) {
        preprocessImage(images[i], d_output + i*batchStride, targetW, targetH, params);
    }
}