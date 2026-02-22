#include "../include/inference_engine.h"
#include "../include/error_helpers.h"
#include <stdio.h>

InferenceEngine:: InferenceEngine(const char* modelPath) : device(torch:kCUDA,0) {
    printf("InferenceEngine constructor called\n");

    try {
        model = torch::jit::load(modelPath);
        model.to(device);
        printf("Model loaded successfully\n");
        model.eval();
        printf("Running on device: %s\n", device.str().c_str());

        inputShape = {1,3,640,640};
        outputShape = {1,25200,85};

        printf("Input shape: %s\n", inputShape.str().c_str());
        printf("Output shape: %s\n", outputShape.str().c_str());

        printf("Do a warmup run to warm up the model\n");
        torch::Tensor dummyInput = torch::randn({1,3,640,640},device);
        torch::NoGradGuard no_grad;
        auto output = model.forward({dummyInput}).toTensor();
        printf("Warmup run completed successfully\n");

        printf("Creating input tensor\n");
        inputTensor = torch::randn({1,3,640,640},device);
        printf("Input tensor created successfully\n");

        printf("Creating output tensor\n");
        outputTensor = torch::randn({1,25200,85},device);
    }
    catch (const c10::Error& e) {
        fprintf(stderr, "Error loading model: %s\n", e.what());
        exit(EXIT_FAILURE);
    }
}

InferenceEngine::~InferenceEngine() {
    // Auto cleans
}

void InferenceEngine::forward(float* d_input, float* d_output) {

    try {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        torch::Tensor inputTensor = torch::from_blob(d_input, {inputShape[0], inputShape[1], inputShape[2], inputShape[3]}, options);

        //Gradient compute disabled for inference
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);

        torch::Tensor output = model.forward(inputs).toTensor();

        if (output.size(0) != outputShape[0] || output.size(1) != outputShape[1] || output.size(2) != outputShape[2]) {
            fprintf(stderr, "Output shape mismatch: %s\n", output.sizes().str().c_str());
         //   exit(EXIT_FAILURE);
        }
        cudaError_t err = cudaMemcpy(d_output, output.data_ptr(), output.numel() * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaError(err, "cudaMemcpy failed");
        printf("Forward pass completed successfully\n");
        
    }
    catch(const c10::Error& e) {
        fprintf(stderr, "Torch error: %s\n", e.what());
        exit(EXIT_FAILURE);
    }
}