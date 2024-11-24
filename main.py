import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog
import os

# Prevent TensorFlow from using the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the saved model
model = tf.keras.models.load_model("skin_cancer_model_tf.h5")
print("Model loaded successfully!")

# Class names
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

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
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png"),  # Supported image formats
            ("JPEG Files", "*.jpg;*.jpeg"),
            ("PNG Files", "*.png"),
            ("All Files", "*.*")  # Option to show all files
        ]
    )
    if file_path:
        display_prediction(file_path)

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

# Function to handle drag-and-drop files
def drop(event):
    file_path = event.data
    if os.path.isfile(file_path):
        display_prediction(file_path)

# Create the GUI
root = TkinterDnD.Tk()
root.title("Skin Cancer Classification")
root.geometry("500x700")

# Frame for drag-and-drop
frame = tk.Frame(root, width=400, height=200, bg="lightgray", relief="ridge", borderwidth=2)
frame.pack(pady=20)
frame.pack_propagate(False)

# Label for drag-and-drop instructions
drag_label = tk.Label(frame, text="Drag and drop an image file here", bg="lightgray", font=("Helvetica", 14))
drag_label.pack(expand=True)

# Bind drag-and-drop event
frame.drop_target_register(DND_FILES)
frame.dnd_bind('<<Drop>>', drop)

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
