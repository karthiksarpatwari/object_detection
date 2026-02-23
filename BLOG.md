# Object Detection Capstone — Notes & Reflections

## Motivation

This project is an exercise in getting into the details of object detection and GPU hardware. I want to explore:

- **Hardware**: Performance on Jetson Nano vs. Thor vs. cloud GPUs (e.g., Blackwell)
- **Deployment**: Running in Colab and cloud environments with powerful GPUs
- **Integration**: How Torch, cuBLAS, cuTENSOR, and cuDNN interplay; how this compares to feeding images to LLMs for detection

The framework forming here can be reused for deeper investigations—video detection, pose estimation (e.g., Google MoveNet), and more.

---

## Torch / LibTorch Observations

### Warm-up Run

A warm-up forward pass is performed during `InferenceEngine` construction. This:

- JIT-compiles the TorchScript graph on first use
- Allocates device memory and caches kernels
- Avoids a large first-frame latency spike during real inference

Without warm-up, the first image would show much higher inference time.

### Forward Pass and Model Configuration

- **Input**: `[1, 3, 640, 640]`, NCHW, float32, ImageNet-normalized
- **Output**: Raw `[1, 25200, 85]` — 85 = (x, y, w, h, obj_conf) + 80 class logits
- Postprocessing (decode, filter, NMS) is done in C++; the model only does the backbone + detection head

### .pt vs. TorchScript

The battle between `.pt` (pickle) and TorchScript took up a lot of debug time:

- **torch.save()** produces Python pickle; C++ `torch::jit::load()` cannot load it
- **torch.jit.trace()** produces TorchScript; this is what LibTorch expects
- Models must be **traced on CPU** if LibTorch is CPU-only, to avoid embedded CUDA device refs

---

## Bug-Driven Observations

During development, several bugs revealed important behavior:

### LibTorch CUDA vs. CPU Builds

`aten::empty_strided` not available for CUDA backend usually means **LibTorch is CPU-only**. The fix: add a CPU fallback—if `model.to(CUDA)` throws, keep the model on CPU and copy tensors between GPU and host during inference. Preprocessing and postprocessing remain on GPU.

### Host vs. Device Pointers in CUDA Kernels

Passing `image->data` (host) into a CUDA kernel that expects device memory causes **illegal memory access**. The image lives in `image->d_data` after `copyImageToDevice()`. Always use `d_data` for kernel input when the data has been copied to the device.

### TorchScript Trace Sanity Checks

YOLOv5’s graph can differ slightly between runs (e.g., batchnorm, internal conditionals). `torch.jit.trace(..., check_trace=True)` then fails with “Graphs differed across invocations.” Using `check_trace=False` allows export; the traced model still runs correctly for inference.

---

## Ideas for Further Research

1. **Video**: Extend to real-time video streams (e.g., webcam, RTSP).
2. **Pose**: Compare with skeleton/pose models (e.g., Google MoveNet) that run on CPU.
3. **LLM vs. CNN**: Feed images to an LLM for detection vs. this pipeline — throughput, latency, cost.
4. **Benchmarking**: Systematic runs on Jetson Nano, T4, A100, Blackwell to compare FPS and power.
