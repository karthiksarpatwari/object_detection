#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

class InferenceEngine {
    public:
        InferenceEngine();
        InferenceEngine(const char* modelPath);
        ~InferenceEngine();

        void forward(float* d_input, float* d_output);

        std::vector<int64_t> getInputShape() const { return inputShape;}
        std::vector<int64_t> getOutputShape() const { return outputShape;}

        int getInputWidth() const { return inputShape[3];}
        int getInputHeight() const { return inputShape[2];}
        int getNumPredictions() const { return outputShape[1];}
        int getPredictionSize() const { return outputShape[2];}
        
    private:
        torch::jit::Module model;
        std::vector<int64_t> inputShape;
        std::vector<int64_t> outputShape;
        torch::Device device;
        bool useCpu_;  // true when CUDA model.to() failed and we fell back to CPU
        float* h_inputBuffer_;   // host buffer for CPU inference (when useCpu_)
        float* h_outputBuffer_;  // host buffer for CPU inference output
};

#endif