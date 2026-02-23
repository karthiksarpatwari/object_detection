#!/usr/bin/env python3

"""Download YOLOv5 models and export to TorchScript for C++ LibTorch inference."""

import argparse
import os
import sys

# YOLOv5 model names supported by torch.hub ultralytics/yolov5
VALID_MODELS = ("yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x")

COCO_CLASSES = [
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
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def download_and_export(model_name: str, output_path: str) -> None:
    """Download YOLOv5 from torch.hub and export to TorchScript for C++ inference."""
    import torch

    print("=" * 50)
    print("YOLOv5 Model Downloader")
    print("=" * 50)
    print(f"Model: {model_name} -> {output_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        print(f"Downloading {model_name} from ultralytics/yolov5...")
        model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
        model.eval()
        print("Model loaded successfully")

        # LibTorch C++ torch::jit::load() requires TorchScript format (not torch.save pickle).
        # Export model.model (DetectionModel) -> raw [1, 25200, 85] for C++ postprocess.
        # Trace on CPU to avoid CUDA device refs (avoids aten::empty_strided CUDA errors).
        raw_model = model.model.cpu()
        dummy_input = torch.rand(1, 3, 640, 640, device="cpu")

        class TraceWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m

            def forward(self, x):
                out = self.model(x)
                return out[0] if isinstance(out, (tuple, list)) else out

        wrapper = TraceWrapper(raw_model)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, dummy_input, check_trace=False)
        traced.save(output_path)
        print(f"TorchScript model saved to {output_path}")

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")

        # Sanity-check inference
        print("\nRunning inference...")
        if os.path.exists("input") and os.listdir("input"):
            sample = os.path.join("input", sorted(os.listdir("input"))[0])
            results = model(sample)
            print(f"Inference on {sample}: {len(results.xyxy[0])} detections")
        else:
            with torch.no_grad():
                model(torch.rand(1, 3, 640, 640))
            print("Inference on dummy input: OK")

        print("Download and export complete.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def save_coco_names(data_dir: str = "data") -> None:
    """Save COCO class names for C++ detector."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "coco.names")
    with open(path, "w") as f:
        f.write("\n".join(COCO_CLASSES))
    print(f"COCO class names saved to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download YOLOv5 model and export to TorchScript for C++ LibTorch inference."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5n",
        choices=VALID_MODELS,
        help="YOLOv5 model variant (e.g. yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/yolov5n.pt",
        help="Output path for TorchScript model (C++ expects this file)",
    )
    parser.add_argument(
        "--no-coco",
        action="store_true",
        help="Skip saving COCO class names to data/coco.names",
    )
    args = parser.parse_args()

    download_and_export(args.model, args.output)
    if not args.no_coco:
        save_coco_names()

    return 0


if __name__ == "__main__":
    sys.exit(main())
