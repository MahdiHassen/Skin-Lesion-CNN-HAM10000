import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# Prevent TensorFlow from using the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the saved model
model = tf.keras.models.load_model("skin_cancer_model_tf.h5")
print("Model loaded successfully!")

# Class names
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# The 7 classes of skin cancer lesions included in this dataset are:
# Melanocytic nevi (nv)
# Melanoma (mel)
# Benign keratosis-like lesions (bkl)
# Basal cell carcinoma (bcc) 
# Actinic keratoses (akiec)
# Vascular lesions (vas)
# Dermatofibroma (df)

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((32, 32))  # Resize to model input size
    img_array = np.asarray(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class of the image
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions)
    return predicted_class_name, confidence

# Function to handle the file selection dialog
def select_file():
    try:
        # Minimal filetypes argument to avoid macOS issues
        file_path = filedialog.askopenfilename(
            filetypes=[("All Files", "*.*")]
        )
        if file_path:
            display_prediction(file_path)
        else:
            print("No file selected.")
    except Exception as e:
        print(f"Error selecting file: {e}")

# Function to display prediction and show image
def display_prediction(image_path):
    # Predict and display the result
    predicted_class_name, confidence = predict_image(image_path)
    result_label.config(text=f"Predicted Class: {predicted_class_name}\nConfidence: {confidence * 100:.2f}%")

    # Display the selected image
    img = Image.open(image_path)
    img.thumbnail((200, 200))  # Resize image for display
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep a reference to avoid garbage collection

# Create the GUI
root = tk.Tk()
root.title("Skin Cancer Classification")
root.geometry("500x700")

# Button to select a file manually
select_button = tk.Button(root, text="Select File", command=select_file, font=("Helvetica", 14))
select_button.pack(pady=20)

# Label to display prediction results
result_label = tk.Label(root, text="", font=("Helvetica", 16), wraplength=450)
result_label.pack(pady=20)

# Label to display the image
image_label = tk.Label(root)
image_label.pack(pady=20)

# Run the GUI event loop
root.mainloop()