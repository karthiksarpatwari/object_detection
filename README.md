# Object Detection with YOLOv5 and Custom CUDA

Advanced CUDA-based object detection capstone project. Uses YOLOv5 with LibTorch C++, custom CUDA kernels for preprocessing/postprocessing, cuBLAS, cuTENSOR, and cuDNN. Supports Google Colab (T4 GPU) and local Linux.

---

## Overview

- **Model**: YOLOv5 (n/s/m/l/x variants)
- **Engine**: LibTorch C++ (`torch::jit::load`)
- **Input**: 640×640, NCHW, normalized
- **Output**: Raw predictions `[1, 25200, 85]` — decoded and filtered in C++

### Pipeline

1. **Preprocessing** (CUDA): Letterbox resize + ImageNet normalization
2. **Inference**: LibTorch (GPU or CPU fallback)
3. **Postprocessing** (CUDA): Decode, filter by confidence
4. **NMS** (host): Non-maximum suppression
5. **Visualization** (CUDA): Draw bounding boxes

---

## Requirements

- **CUDA** 12.x (12.6+ for Colab T4)
- **cuBLAS, cuDNN** (typically bundled with CUDA)
- **cuTENSOR** (`libcutensor-dev` on Ubuntu/Colab)
- **LibTorch** (CPU or CUDA build; see setup)
- **Python 3.8+** with PyTorch (for model download only)

---

## Quick Start (Google Colab)

1. Open a Colab notebook with GPU: **Runtime → Change runtime type → T4 GPU**
2. Run the setup script:

```python
# In a Colab cell:
%run runGoogleColab.py
```

Or manually:

```bash
git clone https://github.com/karthiksarpatwari/object_detection.git
cd object_detection
make colab-setup   # Downloads LibTorch, model, coco.names
make test-images   # Optional: generate 10 test images
make all
make run
```

---

## Local Setup (Linux)

### 1. Install Dependencies

```bash
# CUDA 12.x (apt)
sudo apt-get install nvidia-cuda-toolkit libcudnn-dev libcutensor-dev

# Or from NVIDIA: CUDA toolkit, cuDNN, cuTENSOR
```

### 2. LibTorch

Download LibTorch for your platform:

- **CUDA 12.6**: [libtorch-cxx11-abi-shared-with-deps-2.6.0+cu126.zip](https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip)
- **CPU only**: [libtorch-cxx11-abi-shared-with-deps-2.6.0+cpu.zip](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip)

```bash
wget <url> -O libtorch.zip
unzip libtorch.zip
# Place libtorch/ in project root, or set LIBTORCH_PATH
```

### 3. Build and Run

```bash
make setup
make download-model   # Downloads YOLOv5n + exports to TorchScript
make test-images      # Optional
make all
make run
```

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make setup` | Create `input/`, `output/`, `models/`, `data/` |
| `make colab-setup` | Setup for Colab: download LibTorch cu126, model |
| `make download-model` | Download YOLOv5n, export TorchScript, save COCO names |
| `make test-images` | Generate 10 sample images into `input/` |
| `make all` | Build `object_detector` binary |
| `make run` | Run on `input/` → `output/` with thresholds 0.25, 0.45 |
| `make clean` | Remove binary |

---

## Usage

```bash
./object_detector <input_dir> <output_dir> <conf_thresh> <iou_thresh>
```

**Examples:**

```bash
./object_detector input output 0.25 0.45
./object_detector ./images ./results 0.5 0.4
```

- **conf_thresh**: Confidence threshold (e.g. 0.25)
- **iou_thresh**: IoU threshold for NMS (e.g. 0.45)

---

## Project Layout

```
object_detection/
├── include/          # Headers
│   ├── object_detector.h
│   ├── inference_engine.h
│   ├── preprocessing_kernel.h
│   ├── postprocessing_kernel.h
│   ├── nms_kernel.h
│   ├── bbox_drawer.h
│   ├── cublas_ops.h, cutensor_ops.h
│   ├── image_loader.h, error_helpers.h
│   └── stb_image.h, stb_image_write.h
├── src/              # CUDA/C++ source
│   ├── main.cu
│   ├── object_detector.cu
│   ├── inference_engine.cu
│   ├── preprocessing_kernel.cu
│   ├── postprocessing_kernel.cu
│   ├── nms_kernel.cu
│   ├── bbox_drawer.cu
│   ├── image_loader.cu
│   ├── cublas_ops.cu
│   └── cutensor_ops.cu
├── scripts/
│   ├── download_model.py   # YOLOv5 → TorchScript export
│   └── generate_test_images.py
├── models/           # yolov5n.pt (TorchScript)
├── data/             # coco.names
├── input/            # Input images
├── output/           # Output images with boxes
├── runGoogleColab.py # Colab setup automation
├── Makefile
├── BLOG.md
└── BUG_EVOLUTION.md  # Bug log and learnings
```

---

## Model Variants

Use `download_model.py` for other YOLOv5 sizes:

```bash
python3 scripts/download_model.py --model yolov5s --output models/yolov5s.pt
```

Then change `main.cu` to load `models/yolov5s.pt` and rebuild.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `aten::empty_strided` CUDA error | LibTorch may be CPU-only. A CPU fallback runs inference on CPU; preprocessing stays on GPU. |
| `Graphs differed across invocations` | Trace sanity check disabled via `check_trace=False` in `download_model.py`. |
| `cudaMemcpy illegal memory access` | Ensure preprocessing uses device pointer `d_data`, not host `data` for image source. |
| `yolov5n.pt not found` | Run `make download-model`. |
| `coco.names not found` | Same as above. |
| `LibTorch not found` | Run `make colab-setup` (Colab) or download LibTorch and set `LIBTORCH_PATH`. |

---

## License

See [LICENSE](LICENSE).
