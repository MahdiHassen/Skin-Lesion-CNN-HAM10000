"""
Skin cancer lesion classification using the HAM10000 dataset.
This script uses TensorFlow to train a Convolutional Neural Network (CNN).

Dataset link:
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #force TensorFlow to use CPU.

import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

#load dataset
SIZE = 32
data_dir = "data/reorganized/"

#create a DataFrame with image paths and labels
skin_df = pd.DataFrame(columns=["image_id", "dx", "path"])
for label in os.listdir(data_dir):
    for img_path in glob(os.path.join(data_dir, label, "*.jpg")):
        skin_df = pd.concat(
            [skin_df, pd.DataFrame({"image_id": [os.path.basename(img_path).split('.')[0]], 
                                    "dx": [label], 
                                    "path": [img_path]})],
            ignore_index=True,
        )

#encode labels into numeric values
le = LabelEncoder()
le.fit(skin_df["dx"]) #so now akiec = 0, bcc = 1, etc...
skin_df["label"] = le.transform(skin_df["dx"])

print("Classes:", list(le.classes_)) #print out classes from numerical list.
print(skin_df.sample(10)) #even with augmentation, nv is more than the others 6 to 1, so ull mostly see that here.

#balance dataset
n_samples = 1200 #With augmentation, each labal now has around 1000 'unique' images. upsampling will just repeat sample images. 
balanced_dfs = []
for label in skin_df["label"].unique(): #iterate thru each label to sample only the amount u need
    df_label = skin_df[skin_df["label"] == label]
    balanced_dfs.append(resample(df_label, replace=True, n_samples=n_samples, random_state=42))

skin_df_balanced = pd.concat(balanced_dfs) #add them all into 1 balanced df

#load and preprocess images
skin_df_balanced["image"] = skin_df_balanced["path"].map(
    lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE)))
)

#normalize pixel values to [0, 1]
X = np.asarray(skin_df_balanced["image"].tolist()) / 255.0
Y = skin_df_balanced["label"].values  # Numeric labels
Y_cat = tf.one_hot(Y, depth=7)  # One-hot encoding for labels. so its either a label or its not.

#split data into train/test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat.numpy(), test_size=0.2, random_state=42)

#build CNN model
#LECTURE 8
#chose sequential cuz simplest and also 1 input, 1 output. Autokeras recommended - thank you ak!
#2D convo layer 32/64/128 filters, 3x3 conv window: feature extraction
#max Pooling 2D: downsamples input w maxing, more efficient.
#bread and butter is fully connected / dense layer: classification
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

#train model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=25, batch_size=32)

#evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

#save model
model.save("skin_cancer_model_tf.h5")
print("Model saved as 'skin_cancer_model_tf.h5'")

#generate figures
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

#save accuracy plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy_plot.png")
plt.close()

#save loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_plot.png")
plt.close()

print("Plots saved as 'accuracy_plot.png' and 'loss_plot.png'")

