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

class ExtractionVisualization(ft.UserControl):
    def _init_(self, extraction_directory, saving_directory, flip_file, create_new_folders):
        self.extraction_directory = extraction_directory
        self.saving_directory = saving_directory
        self.flip_file = flip_file
        self.create_new_folders = create_new_folders

    def build(self):
        self.image = ft.Image()
        self.image.width = 420
        self.image.height = 280
        self.cap = None
        self.left_hand_tracing_points_pos = np.repeat((np.zeros(18)), 5)
        self.right_hand_tracing_points_pos = np.repeat((np.zeros(18)), 5)
        return self.image

    def did_mount(self):
        self.th = threading.Thread(target=self.update_timer, args=(), daemon=True)
        self.th.start()

    def will_unmount(self):
        self.th.join()

    def update_timer(self):
        self.begin_extraction()

    
    def draw_landmarks(self, image, hand_results, face_results):
        """Method to draw landmarks on face and hands.

        Keyword arguments:\n
        image -- image that cap has read.\n
        hand_results -- results of MediaPipe Hands model processing.\n
        face_results -- results of MediaPipe FaceMesh model processing.
        """
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

    def extract_results_landmarks(self, hand_results, face_results):
        """"Method to extract landmarks of face and hands.

        Keyword arguments:\n
        hand_results -- results of MediaPipe Hands model processing.\n
        face_results -- results of MediaPipe FaceMesh model processing.
        """
        left_hand_points_pos = np.concatenate([np.zeros(63), np.repeat(np.zeros(18), 5)]) # Array of left hand landmarks and tracing landmarks
        right_hand_points_pos = np.concatenate([np.zeros(63), np.repeat(np.zeros(18), 5)]) # Array of right hand landmarks and tracing landmarks
        face_points_pos = np.zeros(1404) # Array of face landmarks.
        tracing_points_index = [0, 4, 8, 12, 16, 20] # Six points (wrist, thumb_tip, index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip)

        # Variables for hand statuses.
        self.right_hand_visible = False
        self.left_hand_visible = False

        # If face detected
        if face_results.multi_face_landmarks:
            face_points_pos = np.array([[point.x, point.y, point.z] for point in face_results.multi_face_landmarks[0].landmark]).flatten()

        # If hands got detected
        # Note: Classification labels are set completely opposite, because in video they recognize it that way 
        if hand_results.multi_hand_landmarks:
            # Set boolean variables based on showed hands
            for hand in hand_results.multi_handedness:
                if(hand.classification[0].label == "Left"): 
                    self.right_hand_visible = True
                if(hand.classification[0].label == "Right"): 
                    self.left_hand_visible = True

            # If left hand got detected
            if(self.left_hand_visible == True):
                tracing_points = []

                # For each index in the array of needed indexes
                for idx in tracing_points_index: 
                    # Get X, Y, Z coords about the indexed landmark and add them to the array.
                    landmark = hand_results.multi_hand_landmarks[0].landmark[idx]
                    tracing_points.append([landmark.x, landmark.y, landmark.z])

                # Replace created array with NumPy array and make it 1D.
                tracing_points = np.array(tracing_points).flatten()

                # Tracing points update for left hand
                self.left_hand_tracing_points_pos = np.roll(self.left_hand_tracing_points_pos, 18) # Shift array to the right by one element
                self.left_hand_tracing_points_pos[:18] = tracing_points # Replace first element (that came from the end of the array) with a new one

                # Combine data
                landmarks_points = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[0].landmark]).flatten() # Get all landmarks points and make them 1D
                left_hand_points_pos = np.concatenate([landmarks_points, self.left_hand_tracing_points_pos]) # Combine both landmarks' and tracing arrays.

            # If left hand didn't get detected
            elif(self.left_hand_visible == False):
                # Tracing points update for left hand
                self.left_hand_tracing_points_pos = np.roll(self.left_hand_tracing_points_pos, 18) # Shift array to the right by one element
                self.left_hand_tracing_points_pos[:18] = np.zeros(18) # Replace first element (that came from the end of the array) with a new one

                # Combine data
                left_hand_points_pos = np.concatenate([np.zeros(63), self.left_hand_tracing_points_pos]) # Combine empty landmarks' and changed tracing arrays.

            # If right hand got detected
            if(self.right_hand_visible == True):
                tracing_points = []
                
                # For each index in the array of needed indexes
                for idx in tracing_points_index:
                    # Get X, Y, Z coords about the indexed landmark and add them to the array.
                    landmark = hand_results.multi_hand_landmarks[0].landmark[idx]
                    tracing_points.append([landmark.x, landmark.y, landmark.z])

                # Replace created array with NumPy array and make it 1D.
                tracing_points = np.array(tracing_points).flatten()

                # Tracing points update for right hand
                self.right_hand_tracing_points_pos = np.roll(self.right_hand_tracing_points_pos, 18) # Shift array to the right by one element
                self.right_hand_tracing_points_pos[:18] = tracing_points # Replace first element (that came from the end of the array) with a new one

                # Combine data
                landmarks_points = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[0].landmark]).flatten() # Get all landmarks points and make them 1D
                right_hand_points_pos = np.concatenate([landmarks_points, self.right_hand_tracing_points_pos]) # Combine both landmarks' and tracing arrays.
            
            # If right hand didn't get detected
            elif(self.right_hand_visible == False):
                # Tracing points update for right hand
                self.right_hand_tracing_points_pos = np.roll(self.right_hand_tracing_points_pos, 18) # Shift array to the right by one element
                self.right_hand_tracing_points_pos[:18] = np.zeros(18) # Replace first element with zeros

                # Combine data
                right_hand_points_pos = np.concatenate([np.zeros(63), self.right_hand_tracing_points_pos]) # Combine empty landmarks' and changed tracing arrays.

        # If no hands detected
        else:
            # Tracing points update for left hand
            self.left_hand_tracing_points_pos = np.roll(self.left_hand_tracing_points_pos, 18) # Shift array to the right by one place
            self.left_hand_tracing_points_pos[:18] = np.zeros(18) # Replace first element (that came from the end of the array) with a new one

            # Tracing points update for right hand
            self.right_hand_tracing_points_pos = np.roll(self.right_hand_tracing_points_pos, 18) # Shift array to the right by one place
            self.right_hand_tracing_points_pos[:18] = np.zeros(18) # Replace first element (that came from the end of the array) with a new one

            # Combine data
            left_hand_points_pos = np.concatenate([np.zeros(63), self.left_hand_tracing_points_pos]) # Combine empty landmarks' and changed tracing arrays.
            right_hand_points_pos = np.concatenate([np.zeros(63), self.right_hand_tracing_points_pos]) # Combine empty landmarks' and changed tracing arrays.

        # Return full data
        return np.concatenate([face_points_pos, left_hand_points_pos, right_hand_points_pos])

    def update_tracing_points(self, hand_tracing_points_data, new_tracing_points, hand_detected, hand_recognized):
            """Method to update tracing points array.\n
            Shifts tracing point array to the right for one element and adds one at the first index.

            Keyword arguments:\n
            hand_tracing_points_data -- MediaPipe Hands landmarks [0, 4, 8, 12, 16, 20] tracing points collection.\n
            new_tracing_points -- MediaPipe Hands new landmarks for tracing points.\n
            hand_detected -- Boolean value to check, if any hand got detected.\n
            hand_recognized -- Boolean value to check, if specific hand got detected.
            """
            if (hand_recognized == True and hand_detected == True):
                hand_tracing_points_data = np.roll(hand_tracing_points_data, 18) 
                hand_tracing_points_data[:18] = new_tracing_points 
            elif (hand_recognized == False and hand_detected == True):
                hand_tracing_points_data = np.roll(hand_tracing_points_data, 18) 
                hand_tracing_points_data[:18] = np.zeros(18) 
            elif (hand_recognized == False and hand_detected == False):
                hand_tracing_points_data = np.roll(hand_tracing_points_data, 18) 
                hand_tracing_points_data[:18] = np.zeros(18) 


    def begin_extraction(self):
        """Method to start extraction process."""
        if(self.create_new_folders == True):
            self.scan_files_create_folders()
        else:
            self.scan_files()
    
    def scan_files_create_folders(self):
        """Method to scan files in the selected directory (with creating folders for each subfolder in the directory)."""
        for folder in os.listdir(self.extraction_directory):
            file_amount = 0
            current_file = 0

            # Get amount of files in the folder
            for file in os.scandir(os.path.join(self.extraction_directory, folder)):
                if file.is_file():
                    file_amount += 1

            # For each file in the directory
            for file in os.listdir(os.path.join(self.extraction_directory, folder)):
                current_file += 1
                file_path = "{}/{}/{}".format(self.extraction_directory, folder, file)
                self.cap = cv2.VideoCapture(filename=file_path)

                data_path = "{}/{}".format(self.saving_directory, folder)
                if(os.path.isdir(data_path) == False):
                    os.mkdir(data_path)

                # Create a mask for the hands and face
                with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.5, max_num_hands=2) as hands, mp_face_mesh.FaceMesh(min_detection_confidence=0.4, max_num_faces=1) as face_mesh:
                    number = 0
                    while True:
                        orig_path = "{}/{}".format(data_path, number)
                        if(os.path.isdir(orig_path) == False):
                            os.mkdir(orig_path)
                            break
                        elif (os.path.isdir(orig_path)):
                            number += 1

                    # Save original video data
                    self.save_landmarks_data_from_file(folder_name=folder, file_number=current_file, files_amount=file_amount, original_data_path=orig_path, flipped_data_path=None, hands_model=hands, face_mesh_model=face_mesh)
                    
                    number = 0
                    if self.flip_file == True:
                        while True:
                            flip_path = "{}/{}".format(data_path, number)
                            if(os.path.isdir(flip_path) == False):
                                os.mkdir(flip_path)
                                break
                            else:
                                number += 1
                        
                        self.cap = cv2.VideoCapture(filename=file_path)
                        # Save flipped video data
                        self.save_landmarks_data_from_file(folder_name=folder, file_number=current_file, files_amount=file_amount, original_data_path=None, flipped_data_path=flip_path, hands_model=hands, face_mesh_model=face_mesh)

    def scan_files(self):
        """Method to scan files in the selected directory (without creating new folders for subfolders in the directory)."""
        current_file = 0
        file_amount = 0
        # Get amount of files in the folder
        for file in os.scandir(self.extraction_directory):
            if file.is_file():
                file_amount += 1

        # For each file in the directory
        for file in os.listdir(self.extraction_directory):
            current_file += 1
            file_path = "{}/{}".format(self.extraction_directory, file)
            self.cap = cv2.VideoCapture(filename=file_path)
            data_path = self.saving_directory

            # Create a mask for the hands and face
            with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.5, max_num_hands=2) as hands, mp_face_mesh.FaceMesh(min_detection_confidence=0.4, max_num_faces=1) as face_mesh:
                number = 0
                while True:
                    orig_path = "{}/{}".format(data_path, number)
                    if(os.path.isdir(orig_path) == False):
                        os.mkdir(orig_path)
                        break
                    elif (os.path.isdir(orig_path)):
                        number += 1

                # Save original video data
                self.save_landmarks_data_from_file(folder_name=file.rsplit('.')[0], file_number=current_file, files_amount=file_amount, original_data_path=orig_path, flipped_data_path=None, hands_model=hands, face_mesh_model=face_mesh)
                
                number = 0
                if self.flip_file == True:
                    while True:
                        flip_path = "{}/{}".format(data_path, number)
                        if(os.path.isdir(flip_path) == False):
                            os.mkdir(flip_path)
                            break
                        else:
                            number += 1
                    
                    self.cap = cv2.VideoCapture(filename=file_path)

                    # Save flipped video data
                    self.save_landmarks_data_from_file(folder_name=file.rsplit('.')[0], file_number=current_file, files_amount=file_amount, original_data_path=None, flipped_data_path=flip_path, hands_model=hands, face_mesh_model=face_mesh)

    def save_landmarks_data_from_file(self, folder_name, file_number, files_amount, original_data_path, flipped_data_path, hands_model, face_mesh_model):
        """Method to load a file from the desired path.

        Keyword arguments:\n
        folder_name -- name of the folder, where file is located.\n
        file_number -- number of the current file.\n
        files_amount -- amount of the files in the folder.\n
        original_data_path -- path to the folder containing original video data (put None, if you wish to pass it).\n
        flipped_data_path -- path to the folder containing flipped video data (put None, if you wish to pass it).\n
        hands_model -- MediaPipe Hands model.\n
        face_mesh_model -- MediaPipe FaceMesh model.
        """
        frame_number = 0

        while self.cap.isOpened():
            ret, image = self.cap.read() # Read file's image (ret - is that a frame or empty file, image - frame data)

            if ret == False: # If video reached its end, leave this loop
                frame_number = 0
                break

            image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)

            if flipped_data_path: # If video is supposed to be flipped (exists path to save the flipped video)
                image = cv2.flip(image, 1)
                cv2.putText(image, "{} - Flipped Video, {} of {}".format(folder_name, file_number, files_amount), (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            elif original_data_path: # If video is supposed to be original (exists path to save the original video)
                cv2.putText(image, "{} - Original Video, {} of {}".format(folder_name, file_number, files_amount), (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert image to BGR for better face and hand tracking
            image.flags.writeable = False # Disable any modifications of the 2D array
            hand_results = hands_model.process(image) # Get data about hands landmarks
            face_results = face_mesh_model.process(image) # Get data about face landmarks
            image.flags.writeable = True # Allows any modifications of the 2D array
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert image to RGB for better appearance

            data = self.extract_results_landmarks(hand_results, face_results) # Extract data about face and hands landmakrs

            if hand_results.multi_hand_landmarks: # If there are hands in the frame detected
            
                if original_data_path:
                    np.save("{}/{}".format(original_data_path, frame_number), data)
                elif flipped_data_path:
                    np.save("{}/{}".format(flipped_data_path, frame_number), data)           
                
                frame_number += 1

            self.draw_landmarks(image, hand_results, face_results) # Draw landmarks on visualization

            if ret == True: # If video still has more frames
                ret, image_arr = cv2.imencode(".png", image)
                image_b64 = base64.b64encode(image_arr)
                self.image.src_base64 = image_b64.decode("utf-8")
                self.image.update()