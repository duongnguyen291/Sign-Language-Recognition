import os
import pickle
import mediapipe as mp 
import cv2 
import matplotlib.pyplot as plt

mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hand.Hands(static_image_mode=True,min_detection_confidence=0.3)

DATA_DIR = r'D:\github\introToAi\data\asl_alphabet_train\train_1000_img'
NUM_IMG = 900
data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    print("Processing in " + dir_)
    cnt = 0
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        if cnt < NUM_IMG: #chi lay sao cho moi chu cai dung 15 anh phu hop 
            data_aux = []
            img = cv2.imread(os.path.join(DATA_DIR,dir_,img_path))
            img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #tạo các điểm mốc
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                cnt = cnt + 1
                for hand_landmarks in results.multi_hand_landmarks:
                    # mp_drawing.draw_landmarks(
                    #     img_rgb, #image to draw
                    #     hand_landmarks,  #model output
                    #     mp_hand.HAND_CONNECTIONS, #hand connection
                    #     mp_drawing_styles.get_default_hand_landmarks_style(),
                    #     mp_drawing_styles.get_default_hand_connections_style()
                    # )
                    for i in range(0,21):
                        x = hand_landmarks.landmark[i].x 
                        y = hand_landmarks.landmark[i].y   
                        data_aux.append(x)
                        data_aux.append(y)
                if len(data_aux) > 42:
                    data_aux = data_aux[:42] 
                data.append(data_aux)
                labels.append(dir_)  
#                 plt.figure()
#                 plt.imshow(img_rgb)
#                 plt.title(dir_)
# plt.show()
# print(labels)
f = open('data.pickle','wb')
pickle.dump({'data':data,'labels':labels},f)
f.close()

