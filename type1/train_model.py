import os
import warnings
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
from PIL import Image
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Dinh nghia cac bien
gestures = {}
gestures_map = {}
gesture_names = {}
for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    gestures[letter] = letter 
    gestures_map[letter] = len(gestures_map)
    gesture_names[len(gesture_names)] = letter
    
image_path = r'D:\github\introToAi\data\asl_alphabet_train\train_200_img'
models_path = 'models/new_models.hdf5'
rgb = True
imageSize = 224


# Ham xu ly anh resize ve 224x224 va chuyen ve numpy array
def process_image(path):
    try:
        img = Image.open(path)
        img = img.resize((imageSize, imageSize))
        img = np.array(img)
        return img
    except Exception as e:
        print(f"Error processing image '{path}': {str(e)}")
        return None
# Xu ly du lieu dau vao
def process_data(X_data, y_data, batch_size=128):
    processed_X_data = []
    processed_y_data = []
    for i in range(0, len(X_data), batch_size):
        batch_X_data = X_data[i:i+batch_size]
        batch_y_data = y_data[i:i+batch_size]
        batch_X_data = np.array(batch_X_data, dtype='float32')
        if rgb:
            pass
        else:
            batch_X_data = np.stack((batch_X_data,) * 3, axis=-1)
        batch_X_data /= 255
        processed_X_data.append(batch_X_data)
        processed_y_data.append(batch_y_data)
    return processed_X_data, processed_y_data

# Ham duuyet thu muc anh dung de train
def walk_file_tree(image_path):
    X_data = []
    y_data = []
    for directory, subdirectories, files in os.walk(image_path):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(directory, file)
                gesture_name = gestures.get(file[0], None)
                if gesture_name is not None:
                    processed_image = process_image(path)
                    if processed_image is not None:
                        y_data.append(gestures_map[gesture_name])
                        X_data.append(processed_image)
            else:
                continue
    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data
def walk_file_tree_folder(image_path):
    X_data = []
    y_data = []
    if os.path.exists(image_path):  # Kiểm tra xem thư mục chứa ảnh tồn tại không
        for directory, subdirectories, files in os.walk(image_path):
            for subdirectory in subdirectories:
                gesture_name = subdirectory
                gesture_label = gestures_map.get(gesture_name)  # Sử dụng get để tránh lỗi KeyError
                if gesture_label is not None:  # Kiểm tra xem có ký hiệu của tên nhận dạng không
                    subdir_path = os.path.join(directory, subdirectory)
                    for file in os.listdir(subdir_path):
                        if not file.startswith('.'):
                            path = os.path.join(subdir_path, file)
                            X_data.append(process_image(path))
                            y_data.append(gesture_label)
    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data

def filter_classes(X_data, y_data, min_samples=2):
    filtered_X = []
    filtered_y = []
    class_counts = {}
    for i, y in enumerate(y_data):
        if y in class_counts:
            class_counts[y] += 1
        else:
            class_counts[y] = 1
    
    for i, (X, y) in enumerate(zip(X_data, y_data)):
        if class_counts[y] >= min_samples:
            filtered_X.append(X)
            filtered_y.append(y)
    
    return filtered_X, filtered_y

# Load du lieu vao X va Y
X_data, y_data = walk_file_tree_folder(image_path)

# Filter out classes with too few samples
X_data, y_data = filter_classes(X_data, y_data)

# Phan chia du lieu train va test theo ty le 80/20
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state=12, stratify=y_data)
# Dat cac checkpoint de luu lai model tot nhat
model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc',
                               min_delta=0,
                               patience=10,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)

# Khoi tao model
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
optimizer1 = optimizers.Adam()
base_model = model1

# Them cac lop ben tren
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc2a')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', name='fc4')(x)

predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Dong bang cac lop duoi, chi train lop ben tren minh them vao
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stopping, model_checkpoint])

# Luu model da train ra file
model.save('models/mymodel.h5')

