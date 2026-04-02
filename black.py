import os
from PIL import Image
import numpy as np
import glob

# Create a folder named 'black' to store the generated black images
output_folder = 'E:\\Datasets\\RGBD\\STERE1000\\test_data\\black'
os.makedirs(output_folder, exist_ok=True)

# Get all image files from the test_images folder
input_folder = r'E:\\Datasets\\RGBD\\STERE1000\\\test_data\\test_images'
image_files = glob.glob(os.path.join(input_folder, '*.*'))  # Modify pattern if needed (e.g., *.jpg, *.png)

# Generate black images with the same number and filenames as the original images
for image_file in image_files:
    # Get the filename (without path)
    image_name = os.path.basename(image_file)
    
    # Create a black image of size 256x256
    black_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    
    # Save the black image to the 'black' folder with the original filename
    black_image.save(os.path.join(output_folder, image_name))
  
print(f"Black images have been generated and saved in '{output_folder}'.")
