import cv2
import mediapipe as mp 
import pickle
import numpy as np
import tkinter as tk
from tkinter import Label, Text, Scrollbar, VERTICAL, END, Button
from PIL import Image, ImageTk
import time

TIME_DELAY = 0.01 
model_dict = pickle.load(open('./models/modelV03.p','rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hand.Hands(static_image_mode=True, min_detection_confidence=0.3)

# label_dict = {chr(65 + i - 1): chr(65 + i - 1) for i in range(1, 27)}
# print(label_dict)

# Tạo từ điển với các nhãn từ 0 đến 9 và từ A đến Z
label_dict = {str(i): str(i) for i in range(10)}
label_dict.update({chr(65 + i): chr(65 + i) for i in range(26)})
print(label_dict)


# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Sign Language Recognition")

# Khung để hiển thị video
video_frame = Label(root)
video_frame.pack(side=tk.LEFT)

# Nhãn để hiển thị ký tự dự đoán
result_label = Label(root, text="", font=("Helvetica", 24))
result_label.pack()

# Khung để hiển thị lịch sử dự đoán
history_frame = tk.Frame(root)
history_frame.pack(side=tk.RIGHT, fill=tk.Y)

history_label = Label(history_frame, text="History:", font=("Helvetica", 18))
history_label.grid(row=0, column=0, sticky="n")

# Thêm Text widget để hiển thị lịch sử
history_text = Text(history_frame, height=20, width=30, wrap=tk.WORD)
history_text.grid(row=1, column=0, sticky="n")
scrollbar = Scrollbar(history_frame, orient=VERTICAL, command=history_text.yview)
scrollbar.grid(row=1, column=1, sticky="ns")
history_text.config(yscrollcommand=scrollbar.set)

# Nút xóa lịch sử
def clear_history():
    history_text.delete('1.0', END)
    history.clear()

clear_button = Button(history_frame, text="Clear History", command=clear_history)
clear_button.grid(row=2, column=0, sticky="n")

# Danh sách lưu trữ lịch sử các từ đã đoán
history = []

# Biến lưu trữ từ dự đoán trước đó
previous_prediction = ""
last_prediction_time = time.time()

def update_frame():
    global previous_prediction, last_prediction_time
    ret, frame = cap.read()
    if not ret:
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    data_aux = []
    x_ = []
    y_ = []
    H, W, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hand.HAND_CONNECTIONS,  # hand connection
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for i in range(0, 21):
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
        
        current_time = time.time()
        if current_time - last_prediction_time >= TIME_DELAY:  # Delay n seconds
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = label_dict.get(prediction[0], "")
            result_label.config(text=predicted_character)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # tạo hiển thị
            cv2.putText(frame, predicted_character, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            
            if predicted_character != previous_prediction and predicted_character:
                history.append(predicted_character)
                history_text.insert(END, predicted_character + ' ')
                previous_prediction = predicted_character
                last_prediction_time = current_time

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.imgtk = imgtk
    video_frame.configure(image=imgtk)
    video_frame.after(10, update_frame)

def on_key_press(event):
    if event.keysym == 'space':
        history.append(' ')
        history_text.insert(END, ' ')

def on_closing():
    cap.release()
    root.destroy()

root.bind('<KeyPress>', on_key_press)
root.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
root.mainloop()
