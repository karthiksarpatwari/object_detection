// Bounding box drawer kernel
#include "../include/bbox_drawer.h"
#include "../include/error_helpers.h"
#include <stdio.h>

static const Color CLASS_COLORS[80] = {
    {230, 25, 75},   {60, 180, 75},   {255, 225, 25},  {0, 130, 200},   {245, 130, 48},
    {145, 30, 180},  {70, 240, 240},  {240, 50, 230},  {210, 245, 60},  {250, 190, 212},
    {0, 128, 128},   {220, 190, 255}, {170, 110, 40},  {255, 250, 200}, {128, 0, 0},
    {170, 255, 195}, {128, 128, 0},   {255, 215, 180}, {0, 0, 128},     {128, 128, 128},
    {255, 105, 180}, {173, 216, 230}, {144, 238, 144}, {255, 192, 203}, {221, 160, 221},
    {176, 224, 230}, {152, 251, 152}, {255, 228, 196}, {70, 130, 180},  {255, 165, 0},
    {106, 90, 205},  {60, 179, 113},  {255, 20, 147},  {0, 191, 255},   {34, 139, 34},
    {255, 140, 0},   {123, 104, 238}, {72, 61, 139},   {0, 206, 209},   {199, 21, 133},
    {25, 25, 112},   {148, 0, 211},   {255, 127, 80},  {64, 224, 208},  {124, 252, 0},
    {210, 105, 30},  {75, 0, 130},    {0, 255, 127},   {218, 112, 214}, {238, 130, 238},
    {50, 205, 50},   {255, 0, 255},   {184, 134, 11},  {32, 178, 170},  {135, 206, 235},
    {220, 20, 60},   {178, 34, 34},   {0, 100, 0},     {128, 0, 128},   {72, 118, 255},
    {139, 69, 19},   {85, 107, 47},   {205, 92, 92},   {46, 139, 87},   {100, 149, 237},
    {128, 128, 0},   {186, 85, 211},  {0, 139, 139},   {238, 232, 170}, {189, 183, 107},
    {255, 99, 71},   {107, 142, 35},  {65, 105, 225},  {139, 0, 139},   {72, 209, 204},
    {210, 180, 140}, {245, 222, 179}, {95, 158, 160},  {176, 196, 222}, {205, 133, 63},
    {138, 43, 226},  {233, 150, 122}, {154, 205, 50},  {147, 112, 219}, {112, 128, 144},
};

__global__ void drawRectangleKernel(unsigned char* image, int width, int height, int x1, int y1, int x2, int y2, int channels, unsigned char r, unsigned char g, unsigned char b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    bool onBorder = false;

    x1 = max(0, min(x1, width - 1));
    y1 = max(0, min(y1, height - 1));
    x2 = max(0, min(x2, width - 1));
    y2 = max(0, min(y2, height - 1));

    if (x >=x1 && x <= x2) {
        if (y >= y1 && y <= y2) {
            onBorder = true;
    } 
} 

    if (y >=y1 && y <= y2) {
        if ((x>= x1 && x < x1+thickness) || (x <= x2 && x > x2-thickness)) {
            onBorder = true;
        }
    }

    if (onBorder) {
            int idx = (y * width + x) * channels;
            image[idx] = r;
            if (channels > 1) { image[idx+1] = g;}
            if (channels > 2) { image[idx+2] = b;}
            image[idx] = (unsigned char)(r * 255.0f);
    }
}

__global__ void drawTextBackgroundKernel(unsigned char* image, int width, int height, int x, int y, int textWidth, int textHeight, unsigned char r, unsigned char g, unsigned char b) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    if (px >= x && px < x + textWidth && py >= y && py < y + textHeight) {
        int idx = (py * width + px) * 3;
        image[idx] = r;
        if (channels > 1) { image[idx+1] = g;}
        if (channels > 2) { image[idx+2] = b;}
    }
}

__host__ Color getColorForClass(int classId){

    return CLASS_COLORS[classId % 80];
}

__host__ void drawBoundingBoxes(Image* image, std::vector<Detection>& detections, const char** classNames) {
    if (image == NULL || detections.empty()) { return; }

    if (!image -> deviceAllocated){
        copyImageToDevice(image);
    }
// Start drawing bounding boxes
   for(size_t i = 0; i < detections.size(); i++) {
    const Detection& det = detections[i];
    int classId = det.class_id;
    Color color = getColorForClass(classId);
    int x1 = det.x1;
    int y1 = det.y1;
    int x2 = det.x2;
    int y2 = det.y2;

    x1 = fmax(0, fmin(x1, image -> width - 1)); 
    y1 = fmax(0, fmin(y1, image -> height - 1));
    x2 = fmax(0, fmin(x2, image -> width - 1));
    y2 = fmax(0, fmin(y2, image -> height - 1));

    int thickness = 2;

    dim3 blockSize(16,16);
    dim3 gridSize((image -> width + 15)/16, (image -> height + 15)/16);

    drawRectangleKernel<<<gridSize, blockSize>>>(image -> d_data, image -> width, image -> height, x1, y1, x2, y2, image -> channels, color.r, color.g, color.b);
    checkCudaErrors(cudaGetLastError(),"Draw Rectangle Kernel failed");
    // Draw text background
    if(classNames != NULL && y1 >= 20) {
        int labelHeight = 20;
        int labelWidth = 100;

        drawTextBackgroundKernel<<<gridSize, blockSize>>>(image -> d_data, image -> width, image -> height, x1, y1-labelHeight, labelWidth, labelHeight, color.r, color.g, color.b);
        checkCudaErrors(cudaGetLastError(),"Draw Text Background Kernel failed");

 
    }
  }

  cudaDeviceSynchronize();
  copyImageToHost(image);
  //
   }