import cv2
import mediapipe as mp 
import pickle
import numpy as np


model_dict = pickle.load(open('./models/modelV03.p','rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hand.Hands(static_image_mode=True,min_detection_confidence=0.3)

label_dict = {}  # Khởi tạo từ điển

# Tạo ánh xạ từ 1 đến 26 (cho các chữ cái từ 'A' đến 'Z')
for i in range(1, 27):
    label_dict[chr(65 + i)] = chr(65 + i)   # Sử dụng hàm chr() để chuyển đổi số thành chữ cái
label_dict['V'] = 'Hello everyone <33'
print(label_dict)

while True:     
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()

    H, W, _ = frame.shape
    frame_flipped = cv2.flip(frame, 0)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                        frame, #image to draw
                        hand_landmarks,  #model output
                        mp_hand.HAND_CONNECTIONS, #hand connection
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(0,21):
                x = hand_landmarks.landmark[i].x 
                y = hand_landmarks.landmark[i].y   
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
        if len(x_) > 42 or len(y_) > 42:
            x_ = x_[0:42]
            y_ = y_[0:42]
        if len(data_aux) > 42: 
            data_aux = data_aux[:42]

        x1 = int(min(x_) * W) + 10
        y1 = int(min(y_) * H) + 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = label_dict[prediction[0]]
        print(predicted_character)


        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0), 4) #tao hien thi
        cv2.putText(frame,predicted_character,(x1,y1 - 20),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),3,
                    cv2.LINE_AA)
    cv2.imshow('frame',frame)
    cv2.waitKey(5)

cap.release()
cv2.destroyAllWindows()