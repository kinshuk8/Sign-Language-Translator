import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define dataset path
DATASET_PATH = "dataset/"
IMG_SIZE = 64  # Resize all images to 64x64

# Prepare data storage
X = []
Y = []
labels = sorted(os.listdir(DATASET_PATH))  # List all folders (A-Z)

for label_idx, label in enumerate(labels):
    label_path = os.path.join(DATASET_PATH, label)
    
    if os.path.isdir(label_path):  # Ensure it's a folder
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            Y.append(label_idx)

# Convert to NumPy arrays
X = np.array(X, dtype="float32") / 255.0  # Normalize
Y = to_categorical(np.array(Y))  # One-hot encode labels

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Save preprocessed data
np.save("model/X_train.npy", X_train)
np.save("model/X_test.npy", X_test)
np.save("model/Y_train.npy", Y_train)
np.save("model/Y_test.npy", Y_test)

print("Preprocessing complete. Data saved.")
