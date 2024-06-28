import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file pickle
with open('/kaggle/working/data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Kiểm tra và xử lý dữ liệu
data = data_dict['data']
labels = data_dict['labels']
for i, sample in enumerate(data):
    if len(sample) != 42:
        print(f"Sample {i + 1} has an invalid size of {len(sample)}")
        print(labels[i])

# Chuyển đổi dữ liệu thành numpy array
data = np.asarray(data)
labels = np.asarray(labels)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Thiết lập tham số cho Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': [ 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
print(" Thực hiện Grid Search")
# Thực hiện Grid Search
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)
print(" In ra bộ tham số tốt nhất")
# In ra bộ tham số tốt nhất
print("Best parameters found: ", grid_search.best_params_)

# Sử dụng các tham số tốt nhất để huấn luyện lại mô hình
print("Sử dụng các tham số tốt nhất để huấn luyện lại mô hình")
best_model = RandomForestClassifier(**grid_search.best_params_)
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
print("HERE 1")
# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with best parameters:", accuracy)
print("HERE 2")
# In ra báo cáo phân loại chi tiết
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)

# Trực quan hóa ma trận nhầm lẫn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Trực quan hóa báo cáo phân loại
report = classification_report(y_test, y_pred, output_dict=True)
precision = [report[label]['precision'] for label in best_model.classes_]
recall = [report[label]['recall'] for label in best_model.classes_]
f1_score = [report[label]['f1-score'] for label in best_model.classes_]

plt.figure(figsize=(12, 6))
plt.bar(best_model.classes_, precision, color='skyblue', label='Precision')
plt.bar(best_model.classes_, recall, color='orange', label='Recall', alpha=0.7)
plt.bar(best_model.classes_, f1_score, color='green', label='F1-score', alpha=0.5)
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1-score by Class')
plt.legend()
plt.show()

# Lưu mô hình đã huấn luyện
with open('modelV05.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)