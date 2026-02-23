# GPU Object Detection Makefile
# Capstone project - YOLOV5 with LibTorch C++ and Custom CUDA kernels
# Supports Google Colab (T4 GPU, CUDA 12.x) and local Linux/Windows

UNAME_S := $(shell uname -s 2>/dev/null || echo Unknown)

# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -std=c++17 -O3 -w

# T4 GPU (Turing) - sm_75
# Compatible with CUDA 12.8 (Colab T4)
CUDA_ARCH = -arch=sm_75

# Paths - override LIBTORCH_PATH for Colab (use ./libtorch after colab-setup)
LIBTORCH_PATH ?= $(if $(filter Linux Darwin,$(UNAME_S)),./libtorch,/usr/local/libtorch)
CUDA_PATH ?= /usr/local/cuda

INCLUDES = -I./include \
	-I"$(LIBTORCH_PATH)/include" \
	-I"$(CUDA_PATH)/include" \
	-I"$(LIBTORCH_PATH)/include/torch/csrc/api/include"

LDFLAGS = -L"$(CUDA_PATH)/lib64" \
	-L"$(LIBTORCH_PATH)/lib" \
	-Wl,-rpath,$(LIBTORCH_PATH)/lib

LIBS = -lcudart -lcublas -lcutensor -lcudnn -ltorch -ltorch_cuda -lc10 -lc10_cuda

# Target: no .exe on Linux/Colab
TARGET = $(if $(filter Linux Darwin,$(UNAME_S)),object_detector,object_detector.exe)

# Python - use python3 on Linux/Colab
PYTHON ?= $(if $(filter Linux Darwin,$(UNAME_S)),python3,python)

SOURCES = src/main.cu \
	src/object_detector.cu \
	src/nms_kernel.cu \
	src/cutensor_ops.cu \
	src/error_helpers.cu \
	src/postprocessing_kernel.cu \
	src/preprocessing_kernel.cu \
	src/cublas_ops.cu \
	src/bbox_drawer.cu \
	src/inference_engine.cu 

HEADERS = include/object_detector.h \
	include/nms_kernel.h \
	include/cutensor_ops.h \
	include/error_helpers.h \
	include/postprocessing_kernel.h \
	include/preprocessing_kernel.h \
	include/cublas_ops.h \
	include/bbox_drawer.h \
	include/inference_engine.h 
	include/stb_image.h \
	include/stb_image_write.h 

all: check-deps $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	@echo "Building $(TARGET)..."
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(INCLUDES) $(SOURCES) $(LDFLAGS) $(LIBS) -o $@
	@echo "Done!"

check-deps:
	@echo "Checking dependencies..."
	@test -f models/yolov5n.pt || (echo "Error: yolov5n.pt not found in models directory"; echo "Run: make download-model"; exit 1)
	@test -f data/coco.names || (echo "Error: coco.names not found in data directory"; echo "Run: make download-model"; exit 1)
	@test -d "$(LIBTORCH_PATH)" || (echo "Error: LibTorch not found at $(LIBTORCH_PATH)"; echo "On Colab run: make colab-setup"; exit 1)
	@echo "Dependencies OK"

download-model:
	@echo "Downloading yolov5n.pt..."
	@$(PYTHON) scripts/download_model.py
	@echo "Done!"

setup:
	@mkdir -p input output models data
	@echo "Setup complete!"

# Google Colab: download LibTorch for CUDA 12.6 (compatible with CUDA 12.8 T4)
LIBTORCH_CUDA = cu126
LIBTORCH_VER = 2.6.0
LIBTORCH_URL = https://download.pytorch.org/libtorch/$(LIBTORCH_CUDA)/libtorch-cxx11-abi-shared-with-deps-$(LIBTORCH_VER)%2B$(LIBTORCH_CUDA).zip

colab-setup: setup
	@echo "Setting up for Google Colab (T4, CUDA 12.x)..."
	@if [ ! -d libtorch ]; then \
		echo "Downloading LibTorch $(LIBTORCH_VER) for $(LIBTORCH_CUDA)..."; \
		wget -q "https://download.pytorch.org/libtorch/$(LIBTORCH_CUDA)/libtorch-cxx11-abi-shared-with-deps-$(LIBTORCH_VER)%2B$(LIBTORCH_CUDA).zip" -O libtorch.zip && \
		unzip -q libtorch.zip && rm libtorch.zip && \
		echo "LibTorch installed to ./libtorch"; \
	else \
		echo "LibTorch already present at ./libtorch"; \
	fi
	@$(MAKE) download-model
	@echo "Colab setup complete. Run: make all"

test-images:
	@echo "Generating test images..."
	@$(PYTHON) scripts/generate_test_images.py 10
	@echo "Done!"

run: $(TARGET)
	@echo "Running $(TARGET)..."
	@./$(TARGET) input output 0.25 0.45
	@echo "Done!"

.PHONY: all check-deps download-model setup colab-setup test-images run clean

clean:
	@echo "Cleaning up..."
	@rm -f $(TARGET)
	@echo "Done!"