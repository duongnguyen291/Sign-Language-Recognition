import os
import numpy as np
from PIL import Image
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Định nghĩa các biến
gestures_map = {chr(65+i): i for i in range(26)}  # Tạo một bản đồ từ chữ cái sang số từ 0 đến 25
image_path = r'/kaggle/input/asl-alphabet/asl_alphabet_train'
models_path = 'models/saved_model.hdf5'
imageSize = 224
batch_size = 32
rgb = True  # Sử dụng hình ảnh màu

def process_image(path):
    img = Image.open(path)
    img = img.resize((imageSize, imageSize))
    img = img.convert('RGB')  # Chuyển đổi về hình ảnh màu (3 kênh)
    img = np.array(img)
    return img

def walk_file_tree(image_path):
    X_data = []
    y_data = []
    for directory, subdirectories, files in os.walk(image_path):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(directory, file)
                gesture_name = file[0]  # Lấy ký tự đầu tiên của tên tệp tin
                y_data.append(gestures_map[gesture_name])
                X_data.append(process_image(path))
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
    return X_data, y_data
# Load dữ liệu
X_data, y_data = walk_file_tree_folder(image_path)

# Phân chia dữ liệu train và test
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=12, stratify=y_data)

# Chuyển đổi dữ liệu về dạng numpy array
X_train = np.array(X_train)
X_test = np.array(X_test)

# Chuẩn hóa dữ liệu
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# One-hot encode nhãn
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

# Tạo generators cho dữ liệu train và test
train_generator = (X_train, y_train)
test_generator = (X_test, y_test)

# Callbacks
model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto', restore_best_weights=True)

# Khởi tạo mô hình
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))

# Thêm lớp fully connected và lớp softmax
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc2a')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', name='fc4')(x)
predictions = Dense(26, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Đóng băng các lớp dưới, chỉ huấn luyện lớp trên cùng
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model
history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])

# Lưu model đã train
model.save('models/mymodel.h5')