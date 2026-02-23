#! /usr/bin/env python3

import os
import sys
import urllib.request
from PIL import Image
import io

def get_input_dir():
    """Return input directory: repo_root/input (works for Colab /content/object_detection/input)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return os.path.join(repo_root, "input")

def download_sample_images(num_images=10, output_dir=None):
    if output_dir is None:
        output_dir = get_input_dir()

    print("=" * 50)
    print("Downloading sample images")
    print("=" * 50)
    print(f"Output directory: {output_dir}")

    successful = 0
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        url = f"https://picsum.photos/200/300?random={i}"
        output_path = os.path.join(output_dir, f"image_{i:03d}.jpg")
        try:
            with urllib.request.urlopen(url) as response:
                img_data = response.read()
            img = Image.open(io.BytesIO(img_data))

            if img.mode == 'RGBA':
                img = img.convert('RGB')
            max_size = (1920,1080)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size,Image.Resampling.LANCZOS)
            
            img.save(output_path,'JPEG',quality=95)
            successful +=1
        except Exception as e:
            print(f"Error downloading image {i}: {e}")
            continue
    print(f"Downloaded {successful} images to {output_dir}")
    return successful

if __name__ == "__main__":
    num = 10
    output_dir = None

    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
        except ValueError:
            print("Usage: generate_test_images.py [num_images] [output_dir]")
            sys.exit(1)
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print(f"Downloading {num} sample images")
    download_sample_images(num, output_dir)