import pandas as pd
import os
import shutil
from glob import glob

# Define paths for the dataset parts
data_dir_part1 = os.path.join(os.getcwd(), "data/HAM10000_images_part_1")
data_dir_part2 = os.path.join(os.getcwd(), "data/HAM10000_images_part_2")
dest_dir = os.path.join(os.getcwd(), "data/reorganized")

# Read metadata
metadata_path = os.path.join(os.getcwd(), "data/HAM10000_metadata.csv")
skin_df = pd.read_csv(metadata_path)

# Combine paths from both parts into one dictionary
image_paths = {
    os.path.splitext(os.path.basename(x))[0]: x
    for x in glob(os.path.join(data_dir_part1, "*.jpg")) + glob(os.path.join(data_dir_part2, "*.jpg"))
}

# Map the image paths to the metadata DataFrame
skin_df['path'] = skin_df['image_id'].map(image_paths.get)

# Create destination directories for each class
for label in skin_df['dx'].unique():
    os.makedirs(os.path.join(dest_dir, label), exist_ok=True)

# Move images into subfolders based on their labels
for _, row in skin_df.iterrows():
    src_path = row['path']
    if src_path is None:
        print(f"Warning: Missing path for image_id: {row['image_id']}")
        continue  # Skip this image
    dest_path = os.path.join(dest_dir, row['dx'], f"{row['image_id']}.jpg")
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
    else:
        print(f"Image not found at path: {src_path}")
