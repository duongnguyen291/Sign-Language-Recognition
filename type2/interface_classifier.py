import cv2
import mediapipe as mp 
cap = cv2.VideoCapture(2)
mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hand.Hands(static_image_mode=True,min_detection_confidence=0.3)

while True:
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
                    frame, #image to draw
                    hand_landmarks,  #model output
                    mp_hand.HAND_CONNECTIONS, #hand connection
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
    cv2.imshow('frame',frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()