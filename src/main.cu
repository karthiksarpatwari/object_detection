#include "../include/object_detector.h"
#include "../include/error_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

bool directory_exists(const char* path) {
    struct stat info;
    return stat(path, &info) == 0 && S_ISDIR(info.st_mode);
}

void printProgress(int current, int total) {
    int bar_width = 70;
    float progress = (float)current / total;
    int pos = bar_width * progress;
    printf("\r[");
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %3d%%", (int)(progress * 100));
    fflush(stdout);
}

std::tuple<std::string, std::string, float, float> parseCommandLineArguments(int argc, char* argv[]) {
    std::string inputDir = "input";
    std::string outputDir = "output";
    float confThresh = 0.25f;
    float iouThresh = 0.45f;

    if (argc >=2 ) {
        inputDir = argv[1];
    }

    if (argc >=3 ) {
        outputDir = argv[2];
    }

    if (argc >=4 ) {
        confThresh = atof(argv[3]);
    }

    if (argc >=5 ) {
        iouThresh = atof(argv[4]);
    }

    return std::make_tuple(inputDir, outputDir, confThresh, iouThresh);
}

int main(int argc, char* argv[]) {
 
    printf("|----------------------------------------|\n");
    printf("| Object Detection using YOLOv8 CUDA Library |\n");
    printf("|----------------------------------------|\n");
    auto [inputDir, outputDir, confThresh, iouThresh] = parseCommandLineArguments(argc, argv);
    printf("|----------------------------------------|\n");
    printf("Configuration:\n");
    printf("| Input Directory: %s\n", inputDir.c_str());
    printf("| Output Directory: %s\n", outputDir.c_str());
    printf("| Confidence Threshold: %f\n", confThresh);
    printf("| IoU Threshold: %f\n", iouThresh);
    printf("|----------------------------------------|\n");

    if (!directory_exists(inputDir.c_str())) {
        fprintf(stderr, "Error: Input directory does not exist: %s\n", inputDir.c_str());
        return EXIT_FAILURE;
    }

    if (!directory_exists(outputDir.c_str())) {
        fprintf(stderr, "Error: Output directory does not exist: %s\n", outputDir.c_str());
        return EXIT_FAILURE;
    }
    try {
        auto [inputDir, outputDir, confThresh, iouThresh] = parseCommandLineArguments(argc, argv);
    } catch (const std::exception& e) {
        return EXIT_FAILURE;
    }

    printf("Loading model and initializing YOLOv8 detector...\n");
    ObjectDetector detector("models/yolov5n.pt", "data/coco.names");

    printf("Loading images from: %s\n". inputDir.c_str());

    if(images.empty()) {
        fprintf(stderr, "Error: No images found in input directory: %s\n", inputDir.c_str());
        return EXIT_FAILURE;
    }

    printf("Found %zu images to process\n", images.size());

    printf("Processing images...\n");
    
    cudaEvent_t globalStart, globalStop;
    cudaEventCreate(&globalStart);
    cudaEventCreate(&globalStop);
    cudaEventRecord(globalStart, 0);

    int totalImages = images.size();
    int processedImages = 0;

    for (size_t i = 0; i < images.size(); ++i) {
        Image* img = images[i];

        std::vector<Detection> detections = detector.detect(img, confThresh, iouThresh);

        totalDetections += detections.size();
        ObjectDetector::Timing timing = detector.getLastTiming();

        printf("\n[%zu/%zu] %s\n", i + 1, images.size(), img->filename);
        printf("Detections: %zu\n", detections.size());
        printf("Time %.2f ms(%.1f FPS\n",timing.total, 1000.0f/timing.total);
        printf(" Breakdown:\n");
        printf("  Preprocessing: %.2f ms (%.1f%%)\n", timing.preprocessing, timing.preprocessing/timing.total*100);
        printf("  Inference: %.2f ms (%.1f%%)\n", timing.inference, timing.inference/timing.total*100);
        printf("  Postprocessing: %.2f ms (%.1f%%)\n", timing.postprocessing, timing.postprocessing/timing.total*100);
        printf(" NMS: %.2f ms (%.1f%%)\n", timing.nms, timing.nms/timing.total*100);

        for(const auto& det : detections) {
           printf("    - %s (%.2f%%) at [%.0f, %.0f, %.0f, %.0f]\n", det.class_name, det.confidence*100, det.x, det.y, det.w, det.h);
        }

        char outputPath[512];
        const char* baseName = strrchr(img->filename, '/');
        if(baseName == NULL) baseName = strchr(img->filename, '\\');
        if(baseName == NULL) baseName = img->filename;
        else baseName++;
        snprintf(outputPath, sizeof(outputPath), "%s/%s", outputDir.c_str(), baseName);
        saveImage(outputPath, img);

        processedImages++;
        printProgress(processedImages, totalImages);
    }

    cudaEventRecord(globalStop, 0);
    cudaEventSynchronize(globalStop);
    float globalTime = 0.0f;
    cudaEventElapsedTime(&globalTime, globalStart, globalStop);
    printf("\nTotal processing time: %.2f ms (%.1f FPS)\n", globalTime, totalImages/globalTime*1000);
    

    printf("\n");
    printf("|----------------------------------------|\n");
    printf("       Performance Report       \n");
    printf("|----------------------------------------|\n");
    printf("| Total images processed: %d\n", processedImages);
    printf("| Total detections: %d\n", totalDetections);
    printf("| Total time: %.2f ms (%.1f FPS)\n", globalTime, totalImages/globalTime*1000);
    printf("Average time per image: %.2f ms\n", globalTime/totalImages);
    printf(" Throughput: %.1f images/s\n", totalImages/globalTime*1000);
    printf(" Average breakdown:\n");
    printf(" Preprocessing: %.2f ms\n",totalTime/images.size())*(lastTiming.preprocessing/lastTiming.total));
    printf(" Inference: %.2f ms\n",totalTime/images.size())*(lastTiming.inference/lastTiming.total));
    printf(" Postprocessing: %.2f ms\n",totalTime/images.size())*(lastTiming.postprocessing/lastTiming.total));
    printf(" NMS: %.2f ms\n",totalTime/images.size())*(lastTiming.nms/lastTiming.total));
    printf("|----------------------------------------|\n");

    cudaEventDestroy(globalStop);
    cudaEventDestroy(globalStart);
    printf("Program completed successfully.\n");
    printf("|----------------------------------------|\n");
    return EXIT_SUCCESS;
}
