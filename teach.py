import mediapipe as mp # Library for hand detection
import cv2 # Library for camera manipulations
import numpy as np # Library for math
import os # Library for operating with the system

mp_hands = mp.solutions.hands # Load the solution from mediapipe library
mp_face_mesh = mp.solutions.face_mesh # Load the solution from mediapipe library
mp_drawing = mp.solutions.drawing_utils # Enabling drawing utilities from MediaPipe library
mp_drawing_styles = mp.solutions.drawing_styles

learningDir = r'C:\Users\akare\Documents\RTU\Bakalaura darbs\sign-translation-prototype\videos_alphabet' # Directory of files to learn from
dataDir = r'C:\Users\akare\Documents\RTU\Bakalaura darbs\sign-translation-prototype\sign_data' # Directory where to store numpy data

exitKeyPressed = False

# Method to draw landmarks on face and hands.
# Parse image (frame) and processed face and hands results
def drawLandmarks(image, face_results, hand_results):
    # Draw face landmarks, if any
    if face_results.multi_face_landmarks:
        for face in face_results.multi_face_landmarks:
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
            mp_drawing.draw_landmarks(
                image = image, 
                landmark_list = hand, 
                connections = mp_hands.HAND_CONNECTIONS, # Draw landmarks and their connections based on image, hand shown 
                landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1), # Customize landmarks
                connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1) # Customize connections
            ) 

# Method to concatenate face and hands landmarks.
# Parse both results variables.
def extractResultsLandmarks(face_results, hand_results):
    face_points_pos = np.zeros(1404)
    first_hand_points_pos = np.zeros(63)
    second_hand_points_pos = np.zeros(63)

    # If face detected (finished)
    if face_results:
        face_points_pos = np.array([[point.x, point.y, point.z] for point in face_results.multi_face_landmarks[0].landmark]).flatten()

    # If hands detected (W.I.P.)
    if hand_results:
        if hand_results.multi_handedness[0].index == 0:
            first_hand_points_pos = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[0].landmark]).flatten()
        elif hand_results.multi_handedness.index == 1:
            second_hand_points_pos = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[1].landmark]).flatten()
        '''else:
            first_hand_points_pos = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[0].landmark]).flatten()
            second_hand_points_pos = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[1].landmark]).flatten()'''
    
    return np.concatenate([face_points_pos, first_hand_points_pos, second_hand_points_pos])

for file in os.listdir(learningDir):
    if exitKeyPressed: # Check if need to break (useful to prevent all recordings from being learned)
        break

    path = "{}/{}".format(learningDir, file)
    cap = cv2.VideoCapture(path) # Get file from the folder

    dataPath = "{}/{}".format(dataDir, file[:-3])
    os.mkdir(dataPath)
    
    # frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Get the frames of the video
    current_frame = 0 # Current frame

    # Create a mask for the hands and face
    with mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.5, max_num_hands=2) as hands, mp_face_mesh.FaceMesh(min_detection_confidence=0.4, max_num_faces=1) as face_mesh:
        
        origPath = "{}/{}".format(dataPath, "orig")
        flipPath = "{}/{}".format(dataPath, "flip")
        os.mkdir(origPath)
        os.mkdir(flipPath)

        # Save original video data
        while cap.isOpened():
            ret, image = cap.read()
            cv2.putText(image, "Module: Teaching", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

            if ret == False: # If video reached its end
                if exitKeyPressed == False:
                    cap = cv2.VideoCapture(path)
                    break
                else:
                    break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(image) 
            face_results = face_mesh.process(image)

            if hand_results:
                #data = extractResultsLandmarks(face_results, hand_results)
                #np.save("{}/{}".format(origPath, current_frame), data)
                current_frame += 1

            drawLandmarks(image, face_results, hand_results)

            if ret == True: # If video still going
                # Open a window with the app
                cv2.imshow('Latvian Sign Language Model Teaching App', image)

            if cv2.waitKey(10) & 0xFF == 27: # If user pressed button
                exitKeyPressed = True
                break


        current_frame = 0

        # Save flipped video data
        while cap.isOpened():

            ret, image = cap.read()
            image = cv2.flip(image, 1)
            cv2.putText(image, "Module: Teaching", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

            if ret == False: # If video reached its end
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(image) 
            face_results = face_mesh.process(image)

            if hand_results:
                #data = extractResultsLandmarks(face_results, hand_results)
                #np.save("{}/{}".format(flipPath, current_frame), data)
                current_frame += 1

            drawLandmarks(image, face_results, hand_results)

            if ret == True: # If video still going
                # Open a window with the app
                cv2.imshow('Latvian Sign Language Model Teaching App', image)

            if cv2.waitKey(10) & 0xFF == 27: # If user pressed button
                exitKeyPressed = True
                break

    # Turn off the webcam and close all program-connected windows
    cap.release()
    cv2.destroyAllWindows()