import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import pandas as pd
from glob import glob

import matplotlib.pyplot as plt

import cv2
import random as rand
import csv

data_dir = "data/reorganized/"
csv = 'data/HAM10000_metadata.csv'
testcsv ='data/test_metadata.csv'

actualcsv ='data/HAM10000_metadata.csv'
originalcsv ='data/HAM10000_original_metadata.csv' #REMEMBER TO ALWAYS SAVE A COPY OF THE ORIGINAL


image_path ='data/test'
input_folder = 'data/reorganized/bcc'
output_folder = 'data/reorganized/bcc/output'


font = {
        # 'weight' : 'bold',
        'size'   : 25}

plt.rc('font', **font)


def PlotData(csvdir): #matplot lib stuff
    skin_df = pd.read_csv(csvdir)
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(111)
    skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_ylabel('Count')
    ax1.set_title('Cell Type');

    
    plt.show()

"""
#used to plot other parts of the data. not used for this 
    # ax2 = fig.add_subplot(222)
    # skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
    # ax2.set_ylabel('Count', size=15)
    # ax2.set_title('Sex');
    # ax3 = fig.add_subplot(223)
    # skin_df['localization'].value_counts().plot(kind='bar')
    # ax3.set_ylabel('Count',size=12)
    # ax3.set_title('Localization')
    # ax4 = fig.add_subplot(224)
    # sample_age = skin_df[pd.notnull(skin_df['age'])]
    # sns.histplot(skin_df["age"], kde=True)
    # ax4.set_title('Age')
    # plt.tight_layout()
"""
#image = cv2.imread(image_path)
#rotate: ROTATION CREATES BLACK BORDERS WHICH TANKED MY ACCURACY, SO DONT USE
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

def zoom_image(image): 
    value = rand.uniform(1.1, 1.25) #zoom values can change here
    h, w = image.shape[:2]
    # Resize the image
    new_h = int(h * value)
    new_w = int(w * value)
    resized_image = cv2.resize(image, (new_w, new_h))

    # Calculate crop region to get the central part
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    cropped_image = resized_image[start_y:start_y + h, start_x:start_x + w]

    return cropped_image

def flip_image(image): # 0: x-axis, 1: y- axis, -1: both, added 2 as none
    value = rand.randint(-1, 2)
    #print(f"value: {value}")
    # Check if value is not 2, then flip the image
    if value != 2:
        flipped_image = cv2.flip(image, value)
    else:
        # If value is 2, keep the image unchanged... maybe this was unnecessary..
        flipped_image = image
    return flipped_image


def writemetadata(file_path, target_image_id, new_image_id):
    df = pd.read_csv(file_path)
    
    # Filter rows where 'image_id' matches the target, thank you dataframe
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
        # add the new row to the DataFrame
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        # Write the updated DataFrame back to the same CSV file
        df.to_csv(file_path, index=False)
        print(f"New row added: {new_row}")
    else:
        print("errorasda")
   
    
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder): #make output folder
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder): #iterates thru all images by identifying it as jpg
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            basename = os.path.splitext(filename)[0] #grab its name
            print("basename: ", basename)
            image_path = os.path.join(input_folder, filename) #nice lil string pointing to file
            print("imagepath: ", image_path)
            image = cv2.imread(image_path) #so cv2 can read it
            if image is not None:
                for i in range(1): # for when u wanna make multiple copies from 1 image
                    #rotated_image = rotate_image(image)
                    flipped_image = flip_image(image)
                    zoomed_image = zoom_image(flipped_image)

                    # Save the image
                    newfilename = f"{basename}_{i+1}.jpg"
                    output_path = os.path.join(output_folder, newfilename)
                    cv2.imwrite(output_path, zoomed_image)
                    print(f"Processed and saved: {output_path}")
                    #now gotta update csv file with new image_id and dx
                    #look at metadata to copy data from the basename (image_id)
                    writemetadata(actualcsv, basename, newfilename)
                print("done.")
            else:
                print(f"Failed to load image: {image_path}")
        else:
            print(f"Skipped non-image file: {newfilename}") #this only happens at the end of the search


#PlotData(originalcsv)
PlotData(actualcsv)
#process_images(input_folder, output_folder)