# Libraries and dependencies
import flet as ft 
import mediapipe as mp 
import numpy as np 
import cv2 
import threading
import os 
import base64 
import winsound 

mp_hands = mp.solutions.hands 
mp_face_mesh = mp.solutions.face_mesh 
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

class ExtractionVisualization(ft.UserControl):
    def _init_(self, extraction_directory: str, saving_directory: str, flip_file: bool, create_new_folders: bool):
        self.extraction_directory = extraction_directory
        self.saving_directory = saving_directory
        self.flip_file = flip_file
        self.create_new_folders = create_new_folders

    def build(self):
        self.image = ft.Image()
        self.image.width = 420
        self.image.height = 280
        self.cap = None
        self.one_hand_tracing_points_amount = 9
        self.left_hand_tracing_points_pos = np.repeat((np.zeros(self.one_hand_tracing_points_amount)), 5)
        self.right_hand_tracing_points_pos = np.repeat((np.zeros(self.one_hand_tracing_points_amount)), 5)
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
                    connections = mp_hands.HAND_CONNECTIONS, 
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
                ) 

    def extract_results_landmarks(self, hand_results, face_results):
        """"Method to extract landmarks of face and hands.

        Keyword arguments:\n
        hand_results -- results of MediaPipe Hands model processing.\n
        face_results -- results of MediaPipe FaceMesh model processing.
        """
        left_hand_points_pos = np.concatenate([np.zeros(63), np.repeat(np.zeros(self.one_hand_tracing_points_amount), 5)]) # Array of left hand landmarks and tracing points
        right_hand_points_pos = np.concatenate([np.zeros(63), np.repeat(np.zeros(self.one_hand_tracing_points_amount), 5)]) # Array of right hand landmarks and tracing points
        face_points_pos = np.zeros(366) # Array of face landmarks. # 1404 - all landmarks, 366 - outer circle, eyebrows, eyes and mouth.
        tracing_points_indexes = [0, 4, 8] # Three hand model points (wrist, thumb_tip, index_finger_tip)
        face_selected_landmarks_indexes = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 215, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 
                                           46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 
                                           285, 295, 282, 283, 276, 336, 296, 334, 293, 300,
                                           33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
                                           362, 398, 384, 385, 386, 387, 388, 466, 253, 249, 390, 373, 374, 380, 381, 382,
                                           185, 40, 39, 37, 0, 267, 269, 270, 409, 375, 321, 405, 314, 17, 84, 181, 91, 146,
                                           80, 81, 82, 13, 312, 311, 310, 445, 318, 402, 317, 14, 87, 178, 95]
        
        # Variables for hand statuses.
        self.right_hand_visible = False
        self.left_hand_visible = False

        # If face detected
        if face_results.multi_face_landmarks:
            landmarks_list = []

            for idx in face_selected_landmarks_indexes: 
                    # Get X, Y, Z coordinates about the indexed landmark and add them to the array.
                    landmark = face_results.multi_face_landmarks[0].landmark[idx]
                    landmarks_list.append([landmark.x, landmark.y, landmark.z])

            face_points_pos = np.array(landmarks_list).flatten() # Flatten the array to make it 1D

        # If hands got detected
        if hand_results.multi_hand_landmarks:
            # Set boolean variables based on showed hands
            for hand in hand_results.multi_handedness:
                if(hand.classification[0].label == "Left"): 
                    self.left_hand_visible = True
                if(hand.classification[0].label == "Right"): 
                    self.right_hand_visible = True

            # If left hand got detected
            if(self.left_hand_visible == True):
                tracing_points = []

                # For each index in the array of needed indexes
                for idx in tracing_points_indexes: 
                    # Get X, Y, Z coordinates about the indexed landmark and add them to the array.
                    landmark = hand_results.multi_hand_landmarks[0].landmark[idx]
                    tracing_points.append([landmark.x, landmark.y, landmark.z])

                # Replace created array with NumPy array and make it 1D.
                tracing_points = np.array(tracing_points).flatten()
                # Tracing points update for left hand
                self.update_left_hand_tracing_points(tracing_points, True)

                # Combine data
                landmarks_points = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[0].landmark]).flatten() # Get all landmarks points and make them 1D
                left_hand_points_pos = np.concatenate([landmarks_points, self.left_hand_tracing_points_pos]) # Combine both landmarks and tracing array.

            # If left hand didn't get detected
            elif(self.left_hand_visible == False):
                # Tracing points update for left hand
                self.update_left_hand_tracing_points(None, False)
                # Combine data
                left_hand_points_pos = np.concatenate([np.zeros(63), self.left_hand_tracing_points_pos]) # Combine empty landmarks and changed tracing array.

            # If right hand got detected
            if(self.right_hand_visible == True):
                tracing_points = []
                
                # For each index in the array of needed indexes
                for idx in tracing_points_indexes:
                    # Get X, Y, Z coordinates about the indexed landmark and add them to the array.
                    landmark = hand_results.multi_hand_landmarks[0].landmark[idx]
                    tracing_points.append([landmark.x, landmark.y, landmark.z])

                # Replace created array with NumPy array and make it 1D.
                tracing_points = np.array(tracing_points).flatten()
                # Tracing points update for right hand
                self.update_right_hand_tracing_points(tracing_points, True)

                # Combine data
                landmarks_points = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[0].landmark]).flatten() # Get all landmarks points and make them 1D
                right_hand_points_pos = np.concatenate([landmarks_points, self.right_hand_tracing_points_pos]) # Combine both landmarks' and tracing arrays.
            
            # If right hand didn't get detected
            elif(self.right_hand_visible == False):
                # Tracing points update for right hand
                self.update_right_hand_tracing_points(None, False)
                # Combine data
                right_hand_points_pos = np.concatenate([np.zeros(63), self.right_hand_tracing_points_pos]) # Combine empty landmarks and changed tracing arrays.

        # If no hands detected
        else:
            # Tracing points update for left hand
            self.update_left_hand_tracing_points(None, False)
            # Tracing points update for right hand
            self.update_right_hand_tracing_points(None, False)

            # Combine data
            left_hand_points_pos = np.concatenate([np.zeros(63), self.left_hand_tracing_points_pos]) # Combine empty landmarks and changed tracing arrays.
            right_hand_points_pos = np.concatenate([np.zeros(63), self.right_hand_tracing_points_pos]) # Combine empty landmarks and changed tracing arrays.

        # Return final data
        return np.concatenate([face_points_pos, left_hand_points_pos, right_hand_points_pos])

    def update_left_hand_tracing_points(self, new_tracing_points, left_hand_detected: bool):
        """Method to update tracing points array for left hand.\n
        Shifts tracing point array to the right for one element and adds one at the first index.

        Keyword arguments:\n
        hand_tracing_points_data -- MediaPipe Hands landmarks [0, 4, 8] tracing points collection.\n
        new_tracing_points -- MediaPipe Hands new landmarks for tracing points.\n
        left_hand_detected -- Boolean value to check, if left hand got detected.
        """

        if (left_hand_detected == True):
            self.left_hand_tracing_points_pos = np.roll(self.left_hand_tracing_points_pos, self.one_hand_tracing_points_amount) 
            self.left_hand_tracing_points_pos[:self.one_hand_tracing_points_amount] = new_tracing_points 
        else:
            self.left_hand_tracing_points_pos = np.roll(self.left_hand_tracing_points_pos, self.one_hand_tracing_points_amount) 
            self.left_hand_tracing_points_pos[:self.one_hand_tracing_points_amount] = np.zeros(self.one_hand_tracing_points_amount) 

    def update_right_hand_tracing_points(self, new_tracing_points, right_hand_detected: bool):
        """Method to update tracing points array for right hand.\n
        Shifts tracing point array to the right for one element and adds one at the first index.

        Keyword arguments:\n
        new_tracing_points -- MediaPipe Hands new landmarks for tracing points.\n
        right_hand_detected -- Boolean value to check, if right hand got detected.
        """

        if (right_hand_detected == True):
            self.right_hand_tracing_points_pos = np.roll(self.right_hand_tracing_points_pos, self.one_hand_tracing_points_amount) 
            self.right_hand_tracing_points_pos[:self.one_hand_tracing_points_amount] = new_tracing_points 
        else :
            self.right_hand_tracing_points_pos = np.roll(self.right_hand_tracing_points_pos, self.one_hand_tracing_points_amount) 
            self.right_hand_tracing_points_pos[:self.one_hand_tracing_points_amount] = np.zeros(self.one_hand_tracing_points_amount) 

    def begin_extraction(self):
        """Method to start extraction process."""
        if(self.create_new_folders == True):
            self.process_files_new_folders()
        else:
            self.process_files_no_new_folders()
    
    def process_files_new_folders(self):
        """Method to process files in the selected directory (with creating folders for each subfolder in the directory)."""

        # For each folder in the extraction directory
        for folder in os.listdir(self.extraction_directory):
            current_file = 0
            file_amount = 0

            # Get amount of files in the folder
            file_amount = self.count_files(os.path.join(self.extraction_directory, folder))

            # For each file in the directory
            for file in os.listdir(os.path.join(self.extraction_directory, folder)):
                current_file += 1 # Increment the counter
                file_path = "{}/{}/{}".format(self.extraction_directory, folder, file) # Get a path for next file
                self.cap = cv2.VideoCapture(filename=file_path) # Play the file by using OpenCV

                data_path = "{}/{}".format(self.saving_directory, folder) # Get a path for saving
                if(os.path.isdir(data_path) == False): # If path doesn't exist
                    os.mkdir(data_path) # Create it

                # Process the video
                self.process_video(folder_name=folder, file_path=file_path, data_path=data_path, current_file_index=current_file, file_amount_num=file_amount)
            
        # Play a notification sound once extraction process is finished    
        winsound.MessageBeep(type=winsound.MB_ICONASTERISK) 
                
    def process_files_no_new_folders(self):
        """Method to process files in the selected directory (without creating new folders for subfolders in the directory)."""
        current_file = 0
        file_amount = 0

        # Get amount of files in the folder
        file_amount = self.count_files(self.extraction_directory)

        # For each file in the directory
        for file in os.listdir(self.extraction_directory):
            current_file += 1 # Increment the counter
            file_path = "{}/{}".format(self.extraction_directory, file) # Get a path for next file
            self.cap = cv2.VideoCapture(filename=file_path) # Play the file by using OpenCV
            
            data_path = self.saving_directory # Get a path for saving

            # Process the video
            self.process_video(folder_name=file.rsplit('.')[0], file_path=file_path, data_path=data_path, current_file_index=current_file, file_amount_num=file_amount)

        # Play notification sound once extraction process is finished
        winsound.MessageBeep(type=winsound.MB_ICONASTERISK)
            
    def process_video(self, folder_name: str, file_path: str, data_path: str, current_file_index: int, file_amount_num: int):
        """Method to process the video. Extracts data from every frame of the video.

        Keyword arguments:\n
        folder_name -- name of the folder, where file is located.\n
        file_path -- path of the file.\n
        data_path -- path where to store extracted data.\n
        current_file_index -- index of the file processing in order.\n
        file_amount_num -- total amount of the files in the folder.
        """
        # Create a mask for the hands and face via MediaPipe libraries
        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.55, max_num_hands=2) as hands, mp_face_mesh.FaceMesh(min_detection_confidence=0.65, max_num_faces=1) as face_mesh:
            current_frame = 0 

            while True:
                orig_path = "{}/{}".format(data_path, current_frame)
                if(os.path.isdir(orig_path) == False):
                    os.mkdir(orig_path)
                    break
                elif (os.path.isdir(orig_path)):
                    current_frame += 1

            # Save original video data
            self.save_landmarks_data_from_file(folder_name=folder_name, file_number=current_file_index, files_amount=file_amount_num, original_data_path=orig_path, flipped_data_path=None, hands_model=hands, face_mesh_model=face_mesh)
            
            # Reset the counter for flipped video
            current_frame = 0

            if self.flip_file == True:
                while True:
                    flip_path = "{}/{}".format(data_path, current_frame)
                    if(os.path.isdir(flip_path) == False):
                        os.mkdir(flip_path)
                        break
                    else:
                        current_frame += 1
                
                self.cap = cv2.VideoCapture(filename=file_path)

                # Save flipped video data
                self.save_landmarks_data_from_file(folder_name=folder_name, file_number=current_file_index, files_amount=file_amount_num, original_data_path=None, flipped_data_path=flip_path, hands_model=hands, face_mesh_model=face_mesh)

    def save_landmarks_data_from_file(self, folder_name: str, file_number: int, files_amount: int, original_data_path: str, flipped_data_path: str, hands_model, face_mesh_model):
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
        self.left_hand_tracing_points_pos = np.repeat((np.zeros(self.one_hand_tracing_points_amount)), 5)
        self.right_hand_tracing_points_pos = np.repeat((np.zeros(self.one_hand_tracing_points_amount)), 5)

        frame_number = 0

        while self.cap.isOpened():
            ret, image = self.cap.read() # Read file's image (ret - is that a frame or empty file, image - frame data)

            if ret == False: # If video reached its end, leave this loop
                frame_number = 0
                break

            image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)

            if flipped_data_path: # If video is supposed to be flipped (exists path to save the flipped video)
                image = cv2.flip(image, 1)
                cv2.putText(image, "{} - Flipped Video, {} of {}".format(folder_name, file_number, files_amount), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            elif original_data_path: # If video is supposed to be original (exists path to save the original video)
                cv2.putText(image, "{} - Original Video, {} of {}".format(folder_name, file_number, files_amount), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert image to RGB for better face and hand tracking
            image.flags.writeable = False # Disable any modifications of the 2D array
            hand_results = hands_model.process(image) # Get data about hands landmarks
            face_results = face_mesh_model.process(image) # Get data about face landmarks
            image.flags.writeable = True # Allows any modifications of the 2D array
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert image to RGB for better appearance

            data = self.extract_results_landmarks(hand_results, face_results) # Extract data about face and hands landmakrs
            
            if original_data_path:
                np.save("{}/{}".format(original_data_path, frame_number), data)
            elif flipped_data_path:
                np.save("{}/{}".format(flipped_data_path, frame_number), data)           
                
            frame_number += 1

            self.draw_landmarks(image, hand_results, face_results) # Draw landmarks on visualization

            if ret == True: # If video still has more frames, convert them to Base64 format (for Flet library usage)
                ret, image_arr = cv2.imencode(".png", image)
                image_b64 = base64.b64encode(image_arr)
                self.image.src_base64 = image_b64.decode("utf-8")
                self.image.update()

    def count_files(self, directory) -> int:
        """Method to count files in the selected directory.
        
        Keyword arguments:\n
        directory -- Directory where files are located and need to be counted.
        """
        total_file_amount = 0

        for file in os.scandir(directory):
            if file.is_file():
                total_file_amount += 1

        return total_file_amount