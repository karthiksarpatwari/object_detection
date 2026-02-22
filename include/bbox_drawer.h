#ifndef BBOX_DRAWER_H
#define BBOX_DRAWER_H

#include "image_loader.h"
#include "error_helpers.h"
#include <cuda_runtime.h>
#include "postprocessing_kernel.h"

struct Color {
    unsigned char r, g, b;
}

__ global__ void drawRectangleKernel(unsigned char* image, int width, int height, int x1, int y1, int x2, int y2, int channels, unsigned char r, unsigned char g, unsigned char b, int thickness);

__global__ void drawTextBackgroundKernel(unsigned char* image, int width, int height, int x, int y, int width, int height, int channels, unsigned char r, unsigned char g, unsigned char b);

//Host functions

__host__ void drawBoundingBoxes(Image* image, std::vector<Detection>& detections, const char** classNames);

__host__ Color getColorForClass(int classId);

#endif
