#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "image_loader.h"
#include "error_helpers.h"
#include "inference_engine.h"
#include "nms_kernel.h"
#include "bbox_drawer.h"
#include "postprocessing_kernel.h"
#include "preprocessing_kernel.h"
#include "cublas_ops.h"
#include "cutensor_ops.h"

// Object detctor class - responsible for loading the model, preprocessing the image, running the inference, and postprocessing the results

class ObjectDetector {
    public:
        ObjectDetector();
        ObjectDetector(const char* modelPath, const char* labelsPath);
        ~ObjectDetector();
        
        // Main detection function - returns the detections for the given image
        std::vector<Detection> detect(Image* image, float confThresh, float iouThresh);

        // All the performance metrics are stored in this struct in milliseconds
        struct Timing {
            float preprocessing;
            float inference;
            float postprocessing;
            float visualization;
            float total;
            float nms;
        };

        const Timing& getLastTiming() const { return lastTiming; }

        void preprocess(Image* image, float* d_input);
        void inference(float* d_input, float* d_output);
        void postprocess(float* d_output, std::vector<Detection>& detections, float confThresh, int imgWidth, int imgHeight);
    
    private:
        InferenceEngine* engine;
        CublasOps* cublasOps;
        CutensorOps* cutensorOps;
        std::vector<std::string> classNames;
        const char** classNamesPtrs;
        int inputWidth;
        int inputHeight;
        Timing lastTiming;
        
        void loadClassNames(const char* labelsPath);
};

#endif