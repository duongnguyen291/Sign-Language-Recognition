# FIVE LITTE MONKEY GROUP
# Sign Language Recognition

This repository is dedicated to the project on Sign Language Recognition using two different approaches: Convolutional Neural Networks (CNN) and Random Forest Classifier. The primary goal of this project is to recognize sign language through live video and compare the accuracy and efficiency of these two approaches with existing state-of-the-art methods.

## Demo 

[https://github.com/duongnguyen291/Sign-Language-Recognition/assets/116486270/98d5b29c-dc0d-42c4-98c3-a1b8e8c9e22c](https://github.com/duongnguyen291/Sign-Language-Recognition/assets/116486270/98d5b29c-dc0d-42c4-98c3-a1b8e8c9e22c)

## Project Structure

```
D:\code\introToAi\introAi
│
├── .gitignore
├── README.md
├── CNN
│   ├── copy_img.py 
│   ├── detection.py
│   ├── models
│   ├── setup.txt
│   ├── src
│   ├── trainModels.py
│   └── train_model.py
│
├── RandomForest
│   ├── ASL.jpg
│   ├── create_dataset.py
│   ├── data.pickle
│   ├── evaluate.py
│   ├── gui.py
│   ├── interface_classifier.py
│   ├── libary.txt
│   ├── models
│   └── train_classifier.py
│
└── src
    └── collecting_images.py
```

## Approaches

### Convolutional Neural Networks (CNN)
CNNs are deep learning algorithms that are particularly effective for image recognition tasks. In this project, we utilize CNNs to process video frames and recognize sign language gestures. 

### Random Forest Classifier
Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes of the individual trees. This approach is used to classify the video frames into different sign language gestures.

## Objectives

1. **Recognition of Sign Language:** Develop models to recognize sign language gestures from live video.
2. **Comparison of Approaches:** Evaluate and compare the accuracy and efficiency of CNN and Random Forest approaches.
3. **State-of-the-Art Comparison:** Compare the performance of our models with existing state-of-the-art methods.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/duongnguyen291/Sign-Language-Recognition
   ```
2. Navigate to the project directory:
   ```bash
   cd Sign-Language-Recognition
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the CNN Model
Navigate to the `CNN` directory and run the detection script:
```bash
cd CNN
python detection.py
```

### Running the Random Forest Model
Navigate to the `RandomForest` directory and run the GUI script:
```bash
cd RandomForest
python gui.py
```

## Results

The results of our experiments, including the accuracy, precision, recall, and F1-score of both models, will be presented and compared in detail in the `results` section of the documentation.

## Contributors
- **Team Members**:
  - Member 1: Nguyễn Đình Dương
  - Member 2: Nguyễn Mỹ Duyên
  - Member 3: Hồ Bảo Thư
  - Member 4: Hà Việt Khánh
  - Member 5: Nguyễn Trọng Minh Phương

## Contributing

We welcome contributions to improve our project. Please create a pull request with a detailed description of your changes.


## Contact

For any inquiries or issues, please contact us at [2901nguyendinhduong@gmail.com](mailto:2901nguyendinhduong@gmail.com).

---

We hope this project aids in the advancement of sign language recognition technology and provides valuable insights into the effectiveness of different machine learning approaches.
```