#! /usr/bin/env python3

import os
import sys
import urllib, requests
from PIL import Image
import io 

def dowload_sample_images(num_images=10):
    print("="*50)
    print("Downloading sample images")
    print("="*50)


    images = []
    successful = 0
    os.makedirs("input",exist_ok=True)

    for i in range(num_images):
        url = f"https://picsum.photos/200/300?random={i}"
        output_path = os.path.join("input", f"image_{i:0.3d}.jpg")
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
    print(f"Downloaded {successful} images successfully")
    return 0

if __name__ == "__main__":
    num = 10
    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid number of images")
            sys.exit(1)
    print(f"Downloading {num} sample images")

    dowload_sample_images()