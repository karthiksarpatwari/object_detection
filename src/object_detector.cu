#include "../include/object_detector.h"
#include "../include/error_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fstream>

ObjectDetector::ObjectDetector(const char* modelPath, const char* labelsPath) {

    engine = new InferenceEngine(modelPath);
    inputWidth = engine->getInputWidth();
    inputHeight = engine->getInputHeight();
    cublasOps = new CublasOps();
    cutensorOps = new CutensorOps();

    loadClassNames(labelsPath);

    memset(&lastTiming,0,sizeof(Timing));

    printf("ObjectDetector initialized successfully\n");
}

ObjectDetector::~ObjectDetector() {
    delete engine;
    delete cublasOps;
    delete cutensorOps;
    if(classNamesPtrs != NULL){ delete[] classNamesPtrs; }
    printf("ObjectDetector destroyed successfully\n");


}

void ObjectDetector::loadClassNames(const char* labelsPath) {
    std::ifstream file(labelsPath);
    if(!file.is_open()) {
        fprintf(stderr, "Error: Failed to open class names file: %s\n", labelsPath);
        classNamesPtrs = NULL;
        return;
    }


    std::string line;
    while(std::getline(file, line)) {
        classNames.push_back(line);
    }
    file.close();

    classNamesPtrs = new const char*[classNames.size()];
    for(size_t i = 0; i < classNames.size(); i++) {
        classNamesPtrs[i] = classNames[i].c_str();
    }
    printf("Loaded %zu class names successfully\n", classNames.size());
}
std::vector<Detection> ObjectDetector::detect(Image* image, float confThresh, float iouThresh) {

    if (image == NULL) {
        fprintf(stderr, "Error: Image is NULL\n");
        return std::vector<Detection>();
    }

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   

   float* d_preprocessed;
   cudaMalloc(&d_preprocessed, inputWidth * inputHeight * 3 * sizeof(float));
   int numPredictions = engine->getNumPredictions();
   int predictionSize = engine->getPredictionSize();
   float* d_predictions;
   cudaMalloc(&d_predictions, numPredictions * predictionSize * sizeof(float));
   float* d_output;
   cudaMalloc(&d_output, numPredictions * sizeof(Detection));

   cudaEventRecord(start);
   preprocess(image, d_preprocessed);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&lastTiming.preprocessing, start, stop);

   cudaEventRecord(start);
   inference(d_preprocessed, d_predictions);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&lastTiming.inference, start, stop);

   std::vector<Detection> filteredDetections;
   cudaEventRecord(start);
   postprocess(d_predictions, filteredDetections, confThresh, image->width, image->height);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&lastTiming.postprocessing, start, stop);

   std::vector<Detection> nmsDetections;
   cudaEventRecord(start);
   nonMaximumSuppression(filteredDetections, nmsDetections, iouThresh);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&lastTiming.nms, start, stop);
   
   for(size_t i=0; i<nmsDetections.size(); i++) {
    int classId = nmsDetections[i].class_id;
    if (classId >=0 && classId < (int)classNames.size()) {
        strncpy(nmsDetections[i].class_name, classNames[classId].c_str(),63);
        nmsDetections[i].class_name[63] = '\0';
    } else {
        sprintf(nmsDetections[i].class_name, "class_%d", classId);
    }
}

cudaEventRecord(start);
drawBoundingBoxes(image, nmsDetections, classNamesPtrs);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&lastTiming.visualization, start, stop);

lastTiming.total = lastTiming.preprocessing + lastTiming.inference + lastTiming.postprocessing + lastTiming.nms + lastTiming.visualization;

cudaFree(d_preprocessed);
cudaFree(d_predictions);
cudaFree(d_output);
cudaEventDestroy(start);
cudaEventDestroy(stop);
return nmsDetections;
}

void ObjectDetector::preprocess(Image* image, float* d_input) {
    preprocessImage(image,d_input,inputWidth,inputHeight);
}

void ObjectDetector::inference(float* d_input, float* d_output) {
    engine->forward(d_input,d_output);
}

void ObjectDetector::postprocess(float* d_output, std::vector<Detection>& detections, float confThresh, int imgWidth, int imgHeight) {
    decodeAndFilterPredictions(d_output, detections, confThresh, imgWidth, imgHeight, engine->getNumPredictions(), engine->getPredictionSize(), inputWidth, inputHeight);
}
