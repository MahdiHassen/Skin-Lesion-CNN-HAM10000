import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

import cv2
import random as rand
import csv

data_dir = "data/reorganized/"
csv = 'data/HAM10000_metadata.csv'
testcsv ='data/test_metadata.csv'


image_path ='data/test'
#trying to simulate working with the actual directories and csv files
input_folder = 'data/reorganized/df'
output_folder = 'data/reorganized/df/output'

def PlotData(csvdir):
    skin_df = pd.read_csv(csvdir)
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(221)
    skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_ylabel('Count')
    ax1.set_title('Cell Type');

    ax2 = fig.add_subplot(222)
    skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
    ax2.set_ylabel('Count', size=15)
    ax2.set_title('Sex');

    ax3 = fig.add_subplot(223)
    skin_df['localization'].value_counts().plot(kind='bar')
    ax3.set_ylabel('Count',size=12)
    ax3.set_title('Localization')


    ax4 = fig.add_subplot(224)
    sample_age = skin_df[pd.notnull(skin_df['age'])]


    sns.histplot(skin_df["age"], kde=True)
    ax4.set_title('Age')

    plt.tight_layout()
    plt.show()

#image = cv2.imread(image_path)
#rotate
def rotate_image(image):
    # Get the image dimensions (height, width)
    (h, w) = image.shape[:2]
    # Define the center of the image for rotation
    center = (w // 2, h // 2)
    # Define the rotation angle and scaling factor
    angle = rand.randint(2, 90) #rotation in degrees
    scale = 1.0
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def flip_image(image): # 0: x-axis, 1: y- axis, -1: both, added 2 as none
    value = rand.randint(-1, 2)
    print(f"value: {value}")
    # Check if value is not 2, then flip the image
    if value != 2:
        flipped_image = cv2.flip(image, value)
    else:
        # If value is 2, keep the image unchanged
        flipped_image = image
    return flipped_image



def writemetadata(file_path, target_image_id, new_image_id):
    df = pd.read_csv(file_path)
    
    # Filter rows where 'image_id' matches the target
    matching_row = df[df['image_id'] == target_image_id]
    
    # Check if the row exists
    if not matching_row.empty:
        # Isolate the required values from the row
        row_values = matching_row.iloc[0]  # Extract the first (and only) matching row
        lesion_id = row_values['lesion_id']
        dx = row_values['dx']
        dx_type = row_values['dx_type']
        age = row_values['age']
        sex = row_values['sex']
        localization = row_values['localization']
        
        new_lesion_id = 'HAM_NEW10K'
        # Create a new row with updated image_id and lesion_id
        new_row = {
            'lesion_id': new_lesion_id,
            'image_id': new_image_id,
            'dx': dx,
            'dx_type': dx_type,
            'age': age,
            'sex': sex,
            'localization': localization
        }
        # Append the new row to the DataFrame
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        # Write the updated DataFrame back to the same CSV file
        df.to_csv(file_path, index=False)
        print(f"New row added: {new_row}")
    else:
        print("No matching row found.")
   
    
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            basename = os.path.splitext(filename)[0]
            print("basename: ", basename)
            image_path = os.path.join(input_folder, filename)
            print("imagepath: ", image_path)
            image = cv2.imread(image_path)
            if image is not None:
                for i in range(1): #leftover for when u wanna make multiple copies from 1 image
                    rotated_image = rotate_image(image)
                    flipped_image = flip_image(rotated_image)

                    # Save the flipped image
                    newfilename = f"{basename}_{i+1}.jpg"
                    output_path = os.path.join(output_folder, newfilename)
                    cv2.imwrite(output_path, flipped_image)
                    print(f"Processed and saved: {output_path}")
                    #now gotta update csv file with new image_id and dx
                    #look at metadata to copy data from the basename (image_id)
                    writemetadata(testcsv, basename, newfilename)

                print("done.")
            else:
                print(f"Failed to load image: {image_path}")
        else:
            print(f"Skipped non-image file: {newfilename}")



def ResampleData():
    data_dir = "data/reorganized/"

    # Create a DataFrame with image paths and labels
    skin_df = pd.DataFrame(columns=["image_id", "dx", "path"])
    for label in os.listdir(data_dir):
        for img_path in glob(os.path.join(data_dir, label, "*.jpg")):
            skin_df = pd.concat(
                [skin_df, pd.DataFrame({"image_id": [os.path.basename(img_path).split('.')[0]], 
                                        "dx": [label], 
                                        "path": [img_path]})],
                ignore_index=True,
            )

    # Encode labels into numeric values
    le = LabelEncoder()
    le.fit(skin_df["dx"])
    skin_df["label"] = le.transform(skin_df["dx"])

    print("Classes:", list(le.classes_)) #print out classes from numerical list.
    #print(skin_df.sample(10)) #printing out 10 random samples
    SIZE = 32
    n_samples = 500
    balanced_dfs = []
    for label in skin_df["label"].unique():
        df_label = skin_df[skin_df["label"] == label]
        balanced_dfs.append(resample(df_label, replace=True, n_samples=n_samples, random_state=42))

    skin_df_balanced = pd.concat(balanced_dfs)

    # Load and preprocess images
    skin_df_balanced["image"] = skin_df_balanced["path"].map(
        lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE)))
    )

    print(skin_df_balanced['label'].value_counts())

    # Normalize pixel values to [0, 1]
    print("done")

#PlotData(csv)
#ResampleData()

process_images(input_folder, output_folder)