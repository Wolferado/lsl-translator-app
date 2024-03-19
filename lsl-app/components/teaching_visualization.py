import flet as ft
import mediapipe as mp # Library for hand detection
import numpy as np # Library for math
import cv2 # Library for camera manipulations
import threading
import os # Library for operating with the system
import base64 # For conversion

mp_hands = mp.solutions.hands # Load the solution from mediapipe library
mp_face_mesh = mp.solutions.face_mesh # Load the solution from mediapipe library
mp_drawing = mp.solutions.drawing_utils # Enabling drawing utilities from MediaPipe library
mp_drawing_styles = mp.solutions.drawing_styles

class TeachingVisualization(ft.UserControl):
    def _init_(self, learning_directory, saving_directory):
        self.learning_directory = learning_directory
        self.saving_directory = saving_directory

    def build(self):
        self.image = ft.Image()
        self.cap = None
        return self.image

    def did_mount(self):
        self.th = threading.Thread(target=self.update_timer, args=(), daemon=True)
        self.th.start()

    def will_unmount(self):
        self.th.join()

    def update_timer(self):
        self.detectHandsAndFace()

    # Method to draw landmarks on face and hands.
    # Parse image (frame) and processed hands and face results
    def drawLandmarks(self, image, hand_results, face_results):
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

    # Method to concatenate hands and face landmarks.
    # Parse both results variables.
    def extractResultsLandmarks(self, hand_results, face_results):
        face_points_pos = np.zeros(1404)
        left_hand_points_pos = np.zeros(63)
        right_hand_points_pos = np.zeros(63)

        # If face detected
        if face_results.multi_face_landmarks:
            face_points_pos = np.array([[point.x, point.y, point.z] for point in face_results.multi_face_landmarks[0].landmark]).flatten()

        # If hands detected
        # Note: Classification labels are set completely opposite, because in video they recognize it that way 
        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_handedness:
                if hand.classification[0].label == "Right": # Left hand
                    left_hand_points_pos = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[0].landmark]).flatten()
                if hand.classification[0].label == "Left": # Right hand
                    right_hand_points_pos = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[0].landmark]).flatten()

        ''' # For testing purposes
        print("Face")
        print(face_points_pos)
        print("Left")
        if left_hand_points_pos.all() != 0:
            print(left_hand_points_pos)
        print("Right")
        if right_hand_points_pos.all() != 0:
            print(right_hand_points_pos)
        '''

        return np.concatenate([face_points_pos, left_hand_points_pos, right_hand_points_pos])

    # Method to detect hands and face
    def detectHandsAndFace(self):
        for file in os.listdir(self.learning_directory):
            filePath = "{}/{}".format(self.learning_directory, file)
            self.cap = cv2.VideoCapture(filename=filePath)

            dataPath = "{}/{}".format(self.saving_directory, file[:-3])
            if(os.path.isdir(dataPath) == False):
                os.mkdir(dataPath)

            current_frame = 0

            # Create a mask for the hands and face
            with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.5, max_num_hands=2) as hands, mp_face_mesh.FaceMesh(min_detection_confidence=0.4, max_num_faces=1) as face_mesh:
                
                origPath = "{}/{}".format(dataPath, "orig")
                flipPath = "{}/{}".format(dataPath, "flip")

                if(os.path.isdir(origPath) == False):
                    os.mkdir(origPath)
                
                if(os.path.isdir(flipPath) == False):
                    os.mkdir(flipPath)

                # Save original video data
                while self.cap.isOpened():
                    ret, image = self.cap.read()
                    image = cv2.flip(image, 1)
                    cv2.putText(image, "Original Video", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                    if ret == False: # If video reached its end
                        if ret == False:
                            self.cap = cv2.VideoCapture(filePath)
                            break
                        else:
                            break

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    hand_results = hands.process(image) 
                    face_results = face_mesh.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    if hand_results.multi_hand_landmarks:
                        data = self.extractResultsLandmarks(hand_results, face_results)
                        np.save("{}/{}".format(origPath, current_frame), data)
                        current_frame += 1

                    self.drawLandmarks(image, hand_results, face_results)

                    if ret == True: # If video still going
                        ret, image_arr = cv2.imencode(".png", image)
                        image_b64 = base64.b64encode(image_arr)
                        self.image.src_base64 = image_b64.decode("utf-8")
                        self.image.update()

                # Save flipped video data
                while self.cap.isOpened():
                    ret, image = self.cap.read()
                    cv2.putText(image, "Flipped Video", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                    if ret == False: # If video reached its end
                        if ret == False:
                            self.cap = cv2.VideoCapture(filePath)
                            break
                        else:
                            break

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    hand_results = hands.process(image) 
                    face_results = face_mesh.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    if hand_results.multi_hand_landmarks:
                        data = self.extractResultsLandmarks(hand_results, face_results)
                        np.save("{}/{}".format(flipPath, current_frame), data)
                        current_frame += 1

                    self.drawLandmarks(image, hand_results, face_results)

                    if ret == True: # If video still going
                        ret, image_arr = cv2.imencode(".png", image)
                        image_b64 = base64.b64encode(image_arr)
                        self.image.src_base64 = image_b64.decode("utf-8")
                        self.image.update()

        self.cap.release()