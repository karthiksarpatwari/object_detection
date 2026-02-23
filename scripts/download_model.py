#! /usr/bin/env python3

import torch
import os
import sys

def main():
    print("="*50)
    print("YOLOv5 Model Downloader")
    print("="*50)

    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "yolov5n.pt")

    try:
        print("Downloading YOLOv5n from ultralytics/yolov5...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        model.eval()

        print("Model loaded successfully")

        torch.save(model, model_save_path)
        print(f"Model saved to {model_save_path}")

        size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")

        # Run inference with the downloaded model
        print("\nRunning inference...")
        if os.path.exists("input") and os.listdir("input"):
            sample_image = os.path.join("input", sorted(os.listdir("input"))[0])
            results = model(sample_image)
            print(f"Inference on {sample_image}: {len(results.xyxy[0])} detections")
        else:
            dummy_input = torch.rand(1, 3, 640, 640)
            with torch.no_grad():
                results = model(dummy_input)
            print("Inference on dummy input: OK")

        print("Model download and inference completed successfully")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\nSaving COCO class names...")

    try:
        os.makedirs("data", exist_ok=True)
        coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        with open('data/coco.names', 'w') as f:
            for class_name in coco_classes:
                f.write(class_name + "\n")
        print("COCO class names downloaded successfully")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    return 0

if __name__ == "__main__":
    main()