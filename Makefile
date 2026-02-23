# GPU Object Detection Makefile
# Capstone project - YOLOV5 with LibTorch C++ and Custom CUDA kernels

#Compiler and flags
NVCC = nvcc

NVCCFLAGS = -std=c++17 -O3 -w 

CUDA_ARCH = -arch=sm_75

LIBTORCH_PATH = /usr/local/libtorch
CUDA_PATH = /usr/local/cuda

INCLUDES = -I./include \
	-I"$(LIBTORCH_PATH)/include" \
	-I"$(CUDA_PATH)/include" \
	-I"$(LIBTORCH_PATH)/include/torch/csrc/api/include"

LDFLAGS = -L"$(CUDA_PATH)/lib64" \
	-L"$(CUDA_PATH)/lib64" \
	-L"$(LIBTORCH_PATH)/lib"

LIBS = -lcudart -lcublas -lcutensor -lcudnn -ltorch -ltorch_cuda -lc10 -lc10_cuda

TARGET = object_detector.exe

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
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SOURCES) $(LDFLAGS) $(LIBS) -o $@
	@echo "Done!"

check-deps:
	@echo "Checking dependencies..."
	@if not exist "models\yolov5n.torchscript" (\
		echo "Error: YOLOv5n.torchscript not found in models directory"; \
		echo "Please download the model using scripts/download_model.py"; \
		exit 1; \
	fi
	@if not exist "data\coco.names" (\
		echo "Error: coco.names not found in data directory"; \
		echo "Please download the class names using scripts/download_model.py"; \
		exit 1; \
	fi
download-model:
	@echo "Downloading YOLOv5n.torchscript..."
	@python scripts/download_model.py
	@echo "Done!"

setup:
	@if not exist  input mkdir input
	@if not exist output mkdir output
	@if not exist models mkdir models
	@if not exist data mkdir data
	@echo "Setup complete!"

test-images:
	@echo "Generating test images..."
	@python scripts/generate_test_images.py 10
	@echo "Done!"

run: $(TARGET)
	@echo "Running $(TARGET)..."
	@./$(TARGET) input output 0.25 0.45
	@echo "Done!"

.PHONY: all check-deps download-model setup test-images run clean

clean:

	@echo "Cleaning up..."
	@del /q $(TARGET) $(OBJECTS)
	@echo "Done!"

OBJECTS = $(SOURCES:.cu=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	