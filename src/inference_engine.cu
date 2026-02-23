#include "../include/inference_engine.h"
#include "../include/error_helpers.h"
#include <stdio.h>
#include <stdlib.h>

InferenceEngine::InferenceEngine(const char* modelPath) : device(torch::kCUDA, 0), useCpu_(false), h_inputBuffer_(nullptr), h_outputBuffer_(nullptr) {
    printf("InferenceEngine constructor called\n");

    inputShape = {1, 3, 640, 640};
    outputShape = {1, 25200, 85};

    try {
        model = torch::jit::load(modelPath);
        model.eval();

        /* Try moving model to CUDA. If LibTorch is CPU-only or CUDA kernels are missing,
         * model.to(device) throws (e.g. "aten::empty_strided" not available for CUDA). */
        try {
            model.to(device);
            printf("Model loaded successfully on CUDA\n");
        } catch (const c10::Error& e) {
            fprintf(stderr, "CUDA not available for model (LibTorch may be CPU-only): %s\n", e.what());
            fprintf(stderr, "Falling back to CPU inference\n");
            device = torch::kCPU;
            useCpu_ = true;
            /* Allocate host buffers for copy GPU<->CPU during forward */
            size_t inputBytes = (size_t)(inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3]) * sizeof(float);
            size_t outputBytes = (size_t)(outputShape[0] * outputShape[1] * outputShape[2]) * sizeof(float);
            h_inputBuffer_ = (float*)malloc(inputBytes);
            h_outputBuffer_ = (float*)malloc(outputBytes);
            if (!h_inputBuffer_ || !h_outputBuffer_) {
                fprintf(stderr, "Failed to allocate host buffers for CPU inference\n");
                exit(EXIT_FAILURE);
            }
            model.to(device);
        }

        printf("Running on device: %s\n", device.str().c_str());
        printf("Input shape: [%ld, %ld, %ld, %ld]\n", inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
        printf("Output shape: [%ld, %ld, %ld]\n", outputShape[0], outputShape[1], outputShape[2]);

        printf("Doing warmup run...\n");
        torch::NoGradGuard no_grad;
        torch::Tensor dummyInput = torch::randn({1, 3, 640, 640}, device);
        auto output = model.forward({dummyInput}).toTensor();
        printf("Warmup run completed successfully\n");
    }
    catch (const c10::Error& e) {
        fprintf(stderr, "Error loading model: %s\n", e.what());
        if (h_inputBuffer_) free(h_inputBuffer_);
        if (h_outputBuffer_) free(h_outputBuffer_);
        exit(EXIT_FAILURE);
    }
}

InferenceEngine::~InferenceEngine() {
    if (h_inputBuffer_) free(h_inputBuffer_);
    if (h_outputBuffer_) free(h_outputBuffer_);
}

void InferenceEngine::forward(float* d_input, float* d_output) {
    try {
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor output;

        if (useCpu_) {
            /* Copy device input to host, run on CPU, copy output back to device */
            size_t inputNumel = (size_t)(inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3]);
            size_t outputNumel = (size_t)(outputShape[0] * outputShape[1] * outputShape[2]);
            cudaError_t err = cudaMemcpy(h_inputBuffer_, d_input, inputNumel * sizeof(float), cudaMemcpyDeviceToHost);
            checkCudaError(err, "cudaMemcpy D2H input failed");

            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            torch::Tensor inputTensor = torch::from_blob(h_inputBuffer_, {inputShape[0], inputShape[1], inputShape[2], inputShape[3]}, options);
            inputs.push_back(inputTensor);

            output = model.forward(inputs).toTensor();

            err = cudaMemcpy(d_output, output.data_ptr(), outputNumel * sizeof(float), cudaMemcpyHostToDevice);
            checkCudaError(err, "cudaMemcpy H2D output failed");
        } else {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
            torch::Tensor inputTensor = torch::from_blob(d_input, {inputShape[0], inputShape[1], inputShape[2], inputShape[3]}, options);
            inputs.push_back(inputTensor);

            output = model.forward(inputs).toTensor();

            cudaError_t err = cudaMemcpy(d_output, output.data_ptr(), output.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
            checkCudaError(err, "cudaMemcpy D2D output failed");
        }

        if (output.size(0) != outputShape[0] || output.size(1) != outputShape[1] || output.size(2) != outputShape[2]) {
            fprintf(stderr, "Output shape mismatch: [%ld, %ld, %ld]\n", (long)output.size(0), (long)output.size(1), (long)output.size(2));
        }
    }
    catch (const c10::Error& e) {
        fprintf(stderr, "Torch error: %s\n", e.what());
        exit(EXIT_FAILURE);
    }
}