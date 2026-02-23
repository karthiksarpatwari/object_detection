#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

typedef struct { 
    unsigned char* data;
    unsigned char* d_data;
    int width;
    int height;
    int channels;
    bool deviceAllocated;
    char filename[512];
} Image;

__host__ Image* loadImage(const char* filename);
__host__ void saveImage(const char* filename, Image* img);
__host__ void freeImage(Image* img);
__host__ std::vector<Image*> loadImagesFromDirectory(const char* directory);
__host__ void allocateDeviceImage(Image* img);
__host__ void freeDeviceImage(Image* img);  
__host__ void copyImageToDevice(Image* img);
__host__ void copyImageToHost(Image* img);


#endif
