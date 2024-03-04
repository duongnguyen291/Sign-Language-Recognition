import cv2
import os
import time
import uuid

IMAGES_PATH = r'D:\github\introToAi\RealTimeObjectDetection\Tensorflow\workspace\images\data'
 
labels = ['A', 'B', 'C', 'D','del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','Nothing', 'O', 'P', 'Q', 'R', 'S','Spacing', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

number_imgs = 29

for label in labels:
    label_path = os.path.join(IMAGES_PATH,label)
    os.makedirs(label_path, exist_ok=True) #create directory if it doesn't exist 

# Open camera capture outside the loop
cap = cv2.VideoCapture(0)

# Capture images for each label
for num,label in enumerate(labels):
    while True:
        ret, frame = cap.read()
        cv2.imshow('Recognize Sign Language', frame)
        
        # Save image with unique filename
        img_name = str(label) + str(num) + ".jpg"
        img_path = os.path.join(IMAGES_PATH, label, img_name)
        cv2.imwrite(img_path, frame)
        time.sleep(0.5)  # Adjust delay as needed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
