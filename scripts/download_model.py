#! /usr/bin/env python3

import torch
import os
import sys

def main():
    print("="*50)
    print("YOLOv5 Model Downloader")
    print("="*50)

    model_name = "yolov5s.pt"
    model_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
    model_path = os.path.join(os.path.dirname(__file__), model_name)

    try:
        print(f"Downloading {model_name} from {model_url}...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n',pretrained=True)
        model.eval()

        print("Model loaded successfully")
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
                traced_script = torch.jit.trace(model,dummy_input)
        
        traced_script.save('models/yolov5n.torchscript')
        print("TorchScript model saved to models/yolov5n.torchscript")

        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")

        print("Model download completed successfully")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Downloading COCO class names")

    try:
        import urllib.request
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

    print("Downloading YOLOv5n.torchscript")
if __name__ == "__main__":
    main()