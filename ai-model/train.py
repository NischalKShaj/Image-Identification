# file for training the models

# importing the required modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image

# function for loading and preprocessing the data sets
def load_image(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)  # Get class names
    print("🔹 Classes found:", class_names)  

    for index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        image_count = 0  # Track number of images loaded

        for image_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, image_name)
            try:
                img = Image.open(img_path).convert("RGB")  # Ensure 3 channels
                img = img.resize((128, 128))
                images.append(np.array(img) / 255.0)
                labels.append(index)
                image_count += 1
            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")

        print(f"✅ Loaded {image_count} images for class '{class_name}'")

    images = np.array(images, dtype=np.float32).reshape(-1, 128, 128, 3)
    labels = np.array(labels, dtype=np.int32)

    return images, labels, class_names


# load the data set
data_dir="dataset"
x,y, class_names=load_image(data_dir)
if x.size==0 or y.size==0:
    raise ValueError("🚨 No images found! Check if 'dataset/' has images in subfolders.")

# build a simple CNN model
model=keras.Sequential([
    layers.Conv2D(32,(3,3),activation="relu", input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

# compiling and training the models
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x,y, epochs=10)

# saving the model
model.save("model/image_model.h5")

print("✅ Model training complete and saved!")