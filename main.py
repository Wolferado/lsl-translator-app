import mediapipe as mp # Library for hand detection
import cv2 # Library for camera manipulations
import numpy as np # Library for math

mp_hands = mp.solutions.hands # Load the solution from mediapipe library
mp_face_mesh = mp.solutions.face_mesh # Load the solution from mediapipe library

# Enable a camera for the input
cap = cv2.VideoCapture(0)

# Create a mask for the hands
with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.5, max_num_hands=2) as hands, mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=1) as face_mesh:
    while cap.isOpened():
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1) # Flip the stream

        cv2.putText(image, "Module: Recognition", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

        hand_results = hands.process(image) 
        face_results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing = mp.solutions.drawing_utils # Enabling drawing utilities from MediaPipe library
        mp_drawing_styles = mp.solutions.drawing_styles

        # Draw face landmarks, if any
        if face_results.multi_face_landmarks:
            for face in face_results.multi_face_landmarks:
                cv2.putText(image, "Face detected", (10,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec = mp_drawing_styles.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                )

        # Draw hand landmarks, if any
        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                cv2.putText(image, "Hand detected", (10,70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
                mp_drawing.draw_landmarks(
                    image = image, 
                    landmark_list = hand, 
                    connections = mp_hands.HAND_CONNECTIONS, # Draw landmarks and their connections based on image, hand shown 
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), # Customize landmarks
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2) # Customize connections
                ) 
            #for hand in hand_results.multi_handedness:
                #print(hand.classification[0].label)

        # Open a window with the app
        cv2.imshow('Latvian Sign Language Detection App', image)
        
        # Exit statement by the Esc key
        if cv2.waitKey(10) & 0xFF == 27:
            break

# Turn off the webcam and close all program-connected windows
cap.release()
cv2.destroyAllWindows()