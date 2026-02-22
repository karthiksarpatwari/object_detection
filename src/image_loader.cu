#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"
#include "../include/image_loader.h"
#include "./include/error_helpers.h"
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>


__host__ Image* loadImage(const char* filename) {
    Image* img = new Image();
    img->data = stbi_load(filename, &img->width, &img->height, &img->channels, 0);

    if(img->data == NULL) {
        fprintf(stderr, "Error: Failed to load image: %s\n", filename);
        delete img;
        return NULL;
    }

    strncpy(img->filename, filename, sizeof(img->filename) - 1);
    img->filename[sizeof(img->filename) - 1] = '\0';
    img->d_data = NULL;
    img->deviceAllocated = false;
    return img;

}

__host__ saveImage(const char* filename, Image* img) {
    if(img == NULL || img->data == NULL) {
        fprintf(stderr, "Error: Failed to save image: %s\n", filename);
        return;
    }
    // Ensure data is on host
    if (img->deviceAllocated && img->d_data != NULL) {
        copyImageToHost(img);
    }

    const char* ext = strrchr(filename, '.');
    if(ext != NULL) {
        if (strcmp(ext,".png") == 0) {
            stbi_write_png(filename,img->width, img->height, img->channels, img->data, img->width*img->channels);

        } else if (strcmp(ext,".jpg") == 0 || strcmp(ext,".jpeg") == 0 ) {
            stbi_write_jpg(filename, img->width, img->height, img->channels, img->data,95);
        }
         else {
            stbi_write_png(filename, img->width, img->height, img->channels, img->data, img->width * img-> channels);
         }

    }
}

__host__ void freeImage(Image* img) {

    if (img->data !=NULL) {
        stbi_image_free(img->data);
        img->data = NULL;
    }

    if (img->deviceAllocated && img->d_data != NULL) {
        freeDeviceImage(img);
    }

    delete img;
}

__host__ std::vector<Image*> loadImagesFromDirectory(const char* directory) {

    std::vector<Image*> images;

    DIR* dir = opendir(directory);
    if (dir==NULL) {
        fprintf(stderr,"Cannot open directory:%s\n",directory);
        return images;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {

        if(strcmp(entry->d_name,".") == 0 || strcmp(entry->d_name,"..") == 0) continue;

        const char* ext = strrchr(entry->d_name, '.');
        if(ext != NULL) {
            if (strcmp(ext,".png") == 0 || strcmp(ext,".jpg") == 0 || strcmp(ext,".jpeg") == 0) {
                char fullPath[512];
                snprintf(fullPath, sizeof(fullPath), "%s/%s", directory, entry->d_name);
                Image* img = loadImage(fullPath);
                if(img != NULL) {
                    images.push_back(img);
                }
            }
        } else {
            fprintf(stderr,"Unsupported file type: %s\n", entry->d_name);
        }
    }

    closedir(dir);
    return images;
}

__host__ void allocateDeviceImage(Image* img) {
    if(img == NULL || img->deviceAllocated) return;

    size_t size = img->width * img->height * img->channels * sizeof(unsigned char);
    cudaError_t err = cudaMalloc((void**)&img->d_data, size);
    checkCudaError(err, "Failed to allocate device image");
    img->deviceAllocated = true;
}

__host__ void copyImageToDevice(Image* img) {
    if(img == NULL || img->data == NULL) return;
    if(!img->deviceAllocated) allocateDeviceImage(img);
    size_t size = img->width * img->height * img->channels * sizeof(unsigned char);
    cudaError_t err = cudaMemcpy(img->d_data, img->data, size, cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy image to device");
}

__host__ void copyImageToHost(Image* img) {
    if(img == NULL || img->d_data == NULL) return;
    if(!img->deviceAllocated) return;
    size_t size = img->width * img->height * img->channels * sizeof(unsigned char);
    cudaError_t err = cudaMemcpy(img->data, img->d_data, size, cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy image to host");
}

__host__ void freeDeviceImage(Image* img) {
    if(img == NULL || !img->deviceAllocated) return;
    cudaError_t err = cudaFree(img->d_data);
    checkCudaError(err, "Failed to free device image");
    img->d_data = NULL;
    img->deviceAllocated = false;
}
