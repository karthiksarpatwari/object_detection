import os
import shutil

# 1-3. Remove existing repository
repo_dir = 'object_detection'
if os.path.exists(repo_dir):
    print(f"Removing existing '{repo_dir}' directory...")
    shutil.rmtree(repo_dir)
    print(f"'{repo_dir}' removed.")
else:
    print(f"'{repo_dir}' directory does not exist. Skipping removal.")

# 4. Clone the 'object_detection' repository
print("\nCloning the 'object_detection' repository...")
!git clone https://github.com/karthiksarpatwari/object_detection.git
print("'object_detection' repository cloned.")

# 5-8. Replace stb_image.h and stb_image_write.h
repo_path = 'object_detection'
stb_image_url = 'https://raw.githubusercontent.com/nothings/stb/master/stb_image.h'
stb_image_write_url = 'https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h'

stb_image_path_include = os.path.join(repo_path, 'include', 'stb_image.h')
stb_image_write_path_include = os.path.join(repo_path, 'include', 'stb_image_write.h')
stb_image_path_third_party = os.path.join(repo_path, 'third_party', 'stb_image.h')
stb_image_write_path_third_party = os.path.join(repo_path, 'third_party', 'stb_image_write.h')

print("\nDownloading and replacing stb_image.h and stb_image_write.h...")
!wget -O {stb_image_path_include} {stb_image_url}
!wget -O {stb_image_path_third_party} {stb_image_url}
!wget -O {stb_image_write_path_include} {stb_image_write_url}
!wget -O {stb_image_write_path_third_party} {stb_image_write_url}
print("Replaced stb_image.h and stb_image_write.h with official versions.")

# 9. Generate test images into object_detection/input (uses script dir to find repo)
print("\nGenerating test images into object_detection/input...")
!python3 object_detection/scripts/generate_test_images.py 10

# 10. Install Ultralytics
print("\nInstalling Ultralytics...")
!pip install Ultralytics
print("Ultralytics installed.")

# 11. Install libcutensor-dev
print("\nAttempting to install cuTENSOR development libraries...")
!apt-get update
!apt-get install -y libcutensor-dev
print("Installation attempt for cuTENSOR completed.")

# 12. Run makefile commands
os.chdir(repo_path)
print("\nRunning make commands in 'object_detection'...")
!make setup
!make colab-setup
!make all
!make run

# Change back to the parent directory
os.chdir('..')

print("\nMakefile in 'object_detection' executed. Consolidated setup complete.")