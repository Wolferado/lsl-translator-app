import mediapipe as mp # Library for hand detection
import cv2 # Library for camera manipulations
import numpy as np # Library for math

mp_hands = mp.solutions.hands # Load the solution from mediapipe library
mp_face_mesh = mp.solutions.face_mesh # Load the solution from mediapipe library

# Enable a camera for the input
cap = cv2.VideoCapture(0)

# Create a mask for the hands
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=2) as hands, mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=1) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Assign color palette
        image = cv2.flip(image, 1) # Flip the stream
        image.flags.writeable = False # Disable writeable flags

        hand_results = hands.process(image) 
        face_results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_drawing = mp.solutions.drawing_utils # Enabling drawing utilities from MediaPipe library
        mp_drawing_styles = mp.solutions.drawing_styles

        if face_results.multi_face_landmarks:
            for face in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face,
                    connections = mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec = None, # mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        # If there are any multi_hand_landmarks
        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image = image, 
                    landmark_list = hand, 
                    connections = mp_hands.HAND_CONNECTIONS, # Draw landmarks and their connections based on image, hand shown 
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=3), # Customize landmarks
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2) # Customize connections
                ) 

        # Open a window to show the hands
        cv2.imshow('Latvian Sign Language Detection App', image)
        
        # Exit statement by the Esc key
        if cv2.waitKey(10) & 0xFF == 27:
            break

# Turn off the webcam and close all program-connected windows
cap.release()
cv2.destroyAllWindows()