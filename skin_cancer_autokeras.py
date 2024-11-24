"""
Skin cancer lesion classification using the HAM10000 dataset.
This script uses TensorFlow to train a Convolutional Neural Network (CNN).

Dataset link:
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# ---------------------------------
# Step 1: Load Dataset Information
# ---------------------------------
SIZE = 32
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

print("Classes:", list(le.classes_))
print(skin_df.sample(10))

# ---------------------------------
# Step 2: Balance Dataset
# ---------------------------------
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

# Normalize pixel values to [0, 1]
X = np.asarray(skin_df_balanced["image"].tolist()) / 255.0
Y = skin_df_balanced["label"].values  # Numeric labels
Y_cat = tf.one_hot(Y, depth=7)  # One-hot encoding for labels

# ---------------------------------
# Step 3: Split Data into Train/Test Sets
# ---------------------------------
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat.numpy(), test_size=0.2, random_state=42)

# ---------------------------------
# Step 4: Build the CNN Model
# ---------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')  # 7 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# ---------------------------------
# Step 5: Train the Model
# ---------------------------------
history = model.fit(x_train, y_train, validation_split=0.2, epochs=30, batch_size=32)

# ---------------------------------
# Step 6: Evaluate the Model
# ---------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# ---------------------------------
# Step 7: Save the Model
# ---------------------------------
model.save("skin_cancer_model_tf.h5")
print("Model saved as 'skin_cancer_model_tf.h5'")

# ---------------------------------
# Step 8: Visualize Training Results
# ---------------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save accuracy plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy_plot.png")
plt.close()

# Save loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_plot.png")
plt.close()

print("Plots saved as 'accuracy_plot.png' and 'loss_plot.png'")

