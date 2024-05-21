from sklearn.metrics import accuracy_score, classification_report
#train classifier
import pickle
#classifier is random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

data_dict = pickle.load(open('/kaggle/input/picketdataset/data.pickle','rb'))
# print(data_dict)
# for data_set in data_dict['data']:
#     print(data_set)
#     cnt = len(data_set)
#     print("The size: ", cnt)
data = data_dict['data']
label = data_dict['labels']
for i, sample in enumerate(data):
    if len(sample) != 42:
        print(f"Sample {i+1} has an invalid size of {len(sample)}")
        print(label[i])
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.2,shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train,y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict,y_test)

print('{}% of samples were classifier correctly !'.format(score*100))


f = open('model2.p','wb')
pickle.dump({'model':model},f)
f.close()

y_pred = model.predict(x_test)

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# In ra báo cáo phân loại chi tiết
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Trực quan hóa ma trận confusion
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Trực quan hóa báo cáo phân loại
report = classification_report(y_test, y_pred, output_dict=True)
precision = [report[label]['precision'] for label in model.classes_]
recall = [report[label]['recall'] for label in model.classes_]
f1_score = [report[label]['f1-score'] for label in model.classes_]

plt.figure(figsize=(12, 6))
plt.bar(model.classes_, precision, color='skyblue', label='Precision')
plt.bar(model.classes_, recall, color='orange', label='Recall', alpha=0.7)
plt.bar(model.classes_, f1_score, color='green', label='F1-score', alpha=0.5)
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1-score by Class')
plt.legend()
plt.show()