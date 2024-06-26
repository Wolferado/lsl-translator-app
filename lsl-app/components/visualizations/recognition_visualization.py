# Libraries and dependencies
import threading
import flet as ft 
import mediapipe as mp 
import cv2 
import base64 
import keras 
import os 
import numpy as np

from components.symbol_library import signs_lib

mp_hands = mp.solutions.hands 
mp_face_mesh = mp.solutions.face_mesh 
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

signs = list(signs_lib.keys())

class RecognitionVisualization(ft.UserControl):
    def build(self):
        self.image = ft.Image()
        self.image.width = 420
        self.image.height = 280
        self.repeated_recognition_times = 0
        self.left_hand_visible = False
        self.right_hand_visible = False
        self.one_hand_tracing_points_coordinates_amount = 9
        self.left_hand_tracing_points_pos = np.repeat((np.zeros(self.one_hand_tracing_points_coordinates_amount)), 5)
        self.right_hand_tracing_points_pos = np.repeat((np.zeros(self.one_hand_tracing_points_coordinates_amount)), 5)

        self.left_hand_detected_icon = ft.Icon(name=ft.icons.BACK_HAND_OUTLINED, color=ft.colors.GREY)
        self.face_detected_icon = ft.Icon(name=ft.icons.TAG_FACES_OUTLINED, color=ft.colors.GREY)
        self.right_hand_detected_icon = ft.Icon(name=ft.icons.FRONT_HAND_OUTLINED, color=ft.colors.GREY)
        
        self.model_threshold = 0.60

        self.icon_row = ft.Row(
            controls=[self.left_hand_detected_icon, self.face_detected_icon, self.right_hand_detected_icon]
        )

        self.clear_btn = ft.FilledTonalButton(
            text="Clear recognized text",
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=2
            ),
            icon=ft.icons.DELETE_OUTLINE_ROUNDED,
            on_click=self.clear_textfield
        )

        self.dropdown_menu = ft.Dropdown(
            value="RNN",
            autofocus=False,
            text_size=16,
            width=150,
            options=[
                ft.dropdown.Option("RNN"),
                ft.dropdown.Option("LSTM"),
                ft.dropdown.Option("GRU"),
            ],
            on_change=self.on_dropdown_change
        )

        self.icon_row_and_control_elements = ft.Row(
            controls=[self.icon_row, self.clear_btn, self.dropdown_menu],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )

        self.text_field = ft.TextField(
            label="Text Recognition",
            read_only=True,
            value=""
        )

        self.sequence = []
        self.model_name = "lstm_model"
        self.model = keras.models.load_model(os.path.join(os.getcwd(), 'lsl-app', 'models', "{}.keras".format(self.model_name)))
        
        self.cap = cv2.VideoCapture(0)

        return ft.Column(
            alignment=ft.alignment.center,
            controls=[self.image, self.icon_row_and_control_elements, self.text_field]
        )
    
    def did_mount(self): 
        self.th = threading.Thread(target=self.update_timer, args=(), daemon=True)
        self.th.start()

    def will_unmount(self):
        self.th.join()
        self.cap.release() # Fixes endless LED on other pages

    def update_timer(self):
        self.detect_hands_and_face()

    def on_dropdown_change(self, e):
        """Method to call when Sign Language recognition model is changed."""

        # Clear all sequence and tracing points variables.
        self.sequence = []
        self.left_hand_tracing_points_pos = np.repeat((np.zeros(self.one_hand_tracing_points_coordinates_amount)), 5)
        self.right_hand_tracing_points_pos = np.repeat((np.zeros(self.one_hand_tracing_points_coordinates_amount)), 5)

        # Load selected model for further sign recognition
        self.model_name = "{}_model".format(str.lower(self.dropdown_menu.value))
        self.model = keras.models.load_model(os.path.join(os.getcwd(), 'lsl-app', 'models', "{}.keras".format(self.model_name)))

        # Clear text field
        self.clear_textfield()

    def clear_textfield(self, e=None):
        """Method to clear the whole text field."""
        self.text_field.value = ""
        self.text_field.update()

    def detect_hands_and_face(self):
        """Method to detect face and hands in the video stream."""

        # Create a mask for the hands
        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.55, max_num_hands=2) as hands, mp_face_mesh.FaceMesh(min_detection_confidence=0.65, max_num_faces=1) as face_mesh:
            while self.cap.isOpened():
                ret, image = self.cap.read()

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1) # Flip the stream
                image.flags.writeable = False # Disables any modifications of the 2D array
                hand_results = hands.process(image) 
                face_results = face_mesh.process(image)
                image.flags.writeable = True # Allows any modifications of the 2D array
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # self.draw_landmarks(image, hand_results, face_results) # Disable once experimentation is finished
                self.process_landmarks(hand_results, face_results)
                self.update_icon_row(hand_results, face_results)

                # Open a window with the app
                ret, image_arr = cv2.imencode(".png", image)
                image_b64 = base64.b64encode(image_arr)
                self.image.src_base64 = image_b64.decode("utf-8")
                self.image.update()
        
    def process_landmarks(self, hand_results, face_results):
        """Method to process landmarks on face and hands and make prediction.

        Keyword arguments:\n
        hand_results -- Processed hand results.\n
        face_results -- Processed face results.
        """

        # Draw hand landmarks, if any
        if hand_results.multi_hand_landmarks:
            # Set boolean variables for icons to false to check every frame
            self.right_hand_visible = False
            self.left_hand_visible = False

            # Set boolean variables based on showed hands
            for hand in hand_results.multi_handedness:
                if(hand.classification[0].label == "Left"): 
                    self.left_hand_visible = True
                if(hand.classification[0].label == "Right"): 
                    self.right_hand_visible = True

        self.sequence.append(self.extract_landmarks(hand_results, face_results)) # Append extracted landmark information to the sequence

        if(len(self.sequence) >= 40): # If sequence contains information about 40 frames and more
            result = self.model.predict(np.expand_dims(self.sequence[-30:], axis=0))[0] # Get result by parsing expanded sequence array (only last 30 frames)
            print(np.expand_dims(self.sequence, axis=0).shape)
            print(np.array(self.sequence).shape) 

            if (signs[np.argmax(result)] == "_"): # Don't add threshold, if it is blank symbol
                print("Prob: {}, symbol #{} - {}".format(round(float(result[np.argmax(result)]), 3), result.argmax(axis=-1), signs_lib[signs[np.argmax(result)]]))
                if(len(self.text_field.value) == 0 or (self.text_field.value[-2:] == ". " and len(self.text_field.value) > 1)):
                    pass
                elif(self.text_field.value[-1:] != "_"): # For continuation
                    self.text_field.value = self.text_field.value + "_"
                elif(self.text_field.value[-2:] == " _" and len(self.text_field.value) > 1): # For ending point
                    self.text_field.value = self.text_field.value[:-2] + '. '
                elif(self.text_field.value[-1:] == "_"):
                    self.text_field.value = self.text_field.value[:-1] + '. '

                self.sequence.clear()
                self.repeated_recognition_times = 0
                self.cleanup_textfield()
            elif (result[np.argmax(result)] >= self.model_threshold): # If letters exceed needed threshold, output it
                print("Prob: {}, symbol #{} - {}".format(round(float(result[np.argmax(result)]), 3), result.argmax(axis=-1), signs_lib[signs[np.argmax(result)]]))
                if (self.text_field.value[-1:] == "_"):
                    self.text_field.value = self.text_field.value[:-1]
                
                if (len(signs_lib[signs[np.argmax(result)]]) == 1):
                    self.text_field.value = self.text_field.value + "{}".format(signs_lib[signs[np.argmax(result)]])
                elif (len(signs_lib[signs[np.argmax(result)]]) > 1 and (self.text_field.value[-1:] == " " or self.text_field.value[-1:] == "_" or len(self.text_field.value) == 0)):
                    self.text_field.value = self.text_field.value + "{} ".format(signs_lib[signs[np.argmax(result)]])
                else:
                    self.text_field.value = self.text_field.value + " {} ".format(signs_lib[signs[np.argmax(result)]])

                self.sequence.clear()
                self.repeated_recognition_times = 0
                self.cleanup_textfield()
            elif (result[np.argmax(result)] < self.model_threshold and self.repeated_recognition_times < 5): # Repeat for 5 times to be sure
                self.repeated_recognition_times += 1
            else: # Otherwise output notification about insufficient probability
                print("Prob: {}, symbol #{} - {}. (low prob., no display).".format(round(float(result[np.argmax(result)]), 3), result.argmax(axis=-1), signs_lib[signs[np.argmax(result)]]))
                self.sequence.clear()
                self.repeated_recognition_times = 0

    def cleanup_textfield(self):
        """Method to clean up textfield value by deleting words in the beggining."""

        if(len(self.text_field.value) > 50):
            while(len(self.text_field.value) > 45):
                words = self.text_field.value.split( )
                self.text_field.value = " ".join(words[1:])
                self.text_field.update()
        else:
            self.text_field.update()

    def update_icon_row(self, hand_results, face_results):
        """Method to update icon row based on the detected face and hands, for user information."""

        # Draw face landmarks, if any
        if face_results.multi_face_landmarks:
            self.face_detected_icon.color = ft.colors.GREEN_ACCENT_700 # Set icon color to green
        else:
            self.face_detected_icon.color = ft.colors.GREY

        # Draw hand landmarks, if any
        if hand_results.multi_hand_landmarks:
            # Set boolean variables for icons to false to check every frame
            self.right_hand_visible = False
            self.left_hand_visible = False

            # Set boolean variables based on showed hands
            for hand in hand_results.multi_handedness:
                if(hand.classification[0].label == "Left"): 
                    self.left_hand_visible = True
                if(hand.classification[0].label == "Right"): 
                    self.right_hand_visible = True

            if(self.left_hand_visible == True):
                self.left_hand_detected_icon.color = ft.colors.GREEN_ACCENT_700
            else:
                self.left_hand_detected_icon.color = ft.colors.GREY

            if(self.right_hand_visible == True):
                self.right_hand_detected_icon.color = ft.colors.GREEN_ACCENT_700
            else:
                self.right_hand_detected_icon.color = ft.colors.GREY

        else:
            self.left_hand_detected_icon.color = ft.colors.GREY
            self.right_hand_detected_icon.color = ft.colors.GREY

        self.icon_row.update()

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
                self.face_detected_icon.color = ft.colors.GREEN_ACCENT_700 # Set icon color to green
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec = mp_drawing_styles.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                )

        # Draw hand landmarks, if any
        if hand_results.multi_hand_landmarks:
            # For each hand in the frame, draw landamrks
            for hand in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image = image, 
                    landmark_list = hand, 
                    connections = mp_hands.HAND_CONNECTIONS, # Draw landmarks and their connections based on image, hand shown 
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
                ) 

    def extract_landmarks(self, hand_results, face_results):
        """"Method to extract landmarks of face and hands.

        Keyword arguments:\n
        hand_results -- results of MediaPipe Hands model processing.\n
        face_results -- results of MediaPipe FaceMesh model processing.
        """
        left_hand_points_pos = np.concatenate([np.zeros(63), np.repeat(np.zeros(self.one_hand_tracing_points_coordinates_amount), 5)]) # Array of left hand landmarks and tracing landmarks
        right_hand_points_pos = np.concatenate([np.zeros(63), np.repeat(np.zeros(self.one_hand_tracing_points_coordinates_amount), 5)]) # Array of right hand landmarks and tracing landmarks
        face_points_pos = np.zeros(366) # Array of face landmarks. # 1404 - all landmarks, 366 - outer circle, eyebrows, eyes and mouth.
        face_selected_landmarks_indexes = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 215, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 
                                           46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 
                                           285, 295, 282, 283, 276, 336, 296, 334, 293, 300,
                                           33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
                                           362, 398, 384, 385, 386, 387, 388, 466, 253, 249, 390, 373, 374, 380, 381, 382,
                                           185, 40, 39, 37, 0, 267, 269, 270, 409, 375, 321, 405, 314, 17, 84, 181, 91, 146,
                                           80, 81, 82, 13, 312, 311, 310, 445, 318, 402, 317, 14, 87, 178, 95]
        tracing_points_indexes = [0, 4, 8] # Three points (wrist, thumb_tip, index_finger_tip)

        # Variables for hand statuses.
        self.right_hand_visible = False
        self.left_hand_visible = False

        # If face detected
        if face_results.multi_face_landmarks:
            landmarks_list = []

            for idx in face_selected_landmarks_indexes: 
                    # Get X, Y, Z coords about the indexed landmark and add them to the array.
                    landmark = face_results.multi_face_landmarks[0].landmark[idx]
                    landmarks_list.append([landmark.x, landmark.y, landmark.z])

            face_points_pos = np.array(landmarks_list).flatten()
            #face_points_pos = np.array([[point.x, point.y, point.z] for point in face_results.multi_face_landmarks[0].landmark]).flatten()

        # If hands got detected
        # Note: Classification labels are set completely opposite, because in video they recognize it that way 
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
                    # Get X, Y, Z coords about the indexed landmark and add them to the array.
                    landmark = hand_results.multi_hand_landmarks[0].landmark[idx]
                    tracing_points.append([landmark.x, landmark.y, landmark.z])

                # Replace created array with NumPy array and make it 1D.
                tracing_points = np.array(tracing_points).flatten()
                # Tracing points update for left hand
                self.update_left_hand_tracing_points(tracing_points, True)

                # Combine data
                landmarks_points = np.array([[point.x, point.y, point.z] for point in hand_results.multi_hand_landmarks[0].landmark]).flatten() # Get all landmarks points and make them 1D
                left_hand_points_pos = np.concatenate([landmarks_points, self.left_hand_tracing_points_pos]) # Combine both landmarks' and tracing arrays.

            # If left hand didn't get detected
            elif(self.left_hand_visible == False):
                # Tracing points update for left hand
                self.update_left_hand_tracing_points(None, False)
                # Combine data
                left_hand_points_pos = np.concatenate([np.zeros(63), self.left_hand_tracing_points_pos]) # Combine empty landmarks' and changed tracing arrays.

            # If right hand got detected
            if(self.right_hand_visible == True):
                tracing_points = []
                
                # For each index in the array of needed indexes
                for idx in tracing_points_indexes:
                    # Get X, Y, Z coords about the indexed landmark and add them to the array.
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
                right_hand_points_pos = np.concatenate([np.zeros(63), self.right_hand_tracing_points_pos]) # Combine empty landmarks' and changed tracing arrays.

        # If no hands detected
        else:
            # Tracing points update for left hand
            self.update_left_hand_tracing_points(None, False)
            # Tracing points update for right hand
            self.update_right_hand_tracing_points(None, False)

            # Combine data
            left_hand_points_pos = np.concatenate([np.zeros(63), self.left_hand_tracing_points_pos]) # Combine empty landmarks' and changed tracing arrays.
            right_hand_points_pos = np.concatenate([np.zeros(63), self.right_hand_tracing_points_pos]) # Combine empty landmarks' and changed tracing arrays.

        # Return full data
        return np.concatenate([face_points_pos, left_hand_points_pos, right_hand_points_pos])
    
    def update_left_hand_tracing_points(self, new_tracing_points, left_hand_detected):
            """Method to update tracing points array for left hand.\n
            Shifts tracing point array to the right for one element and adds one at the first index.

            Keyword arguments:\n
            hand_tracing_points_data -- MediaPipe Hands landmarks [0, 4, 8] tracing points collection.\n
            new_tracing_points -- MediaPipe Hands new landmarks for tracing points.\n
            left_hand_detected -- Boolean value to check, if left hand got detected.
            """

            if (left_hand_detected == True):
                self.left_hand_tracing_points_pos = np.roll(self.left_hand_tracing_points_pos, self.one_hand_tracing_points_coordinates_amount) 
                self.left_hand_tracing_points_pos[:self.one_hand_tracing_points_coordinates_amount] = new_tracing_points 
            else:
                self.left_hand_tracing_points_pos = np.roll(self.left_hand_tracing_points_pos, self.one_hand_tracing_points_coordinates_amount) 
                self.left_hand_tracing_points_pos[:self.one_hand_tracing_points_coordinates_amount] = np.zeros(self.one_hand_tracing_points_coordinates_amount) 

    def update_right_hand_tracing_points(self, new_tracing_points, right_hand_detected):
            """Method to update tracing points array for right hand.\n
            Shifts tracing point array to the right for one element and adds one at the first index.

            Keyword arguments:\n
            hand_tracing_points_data -- MediaPipe Hands landmarks [0, 4, 8] tracing points collection.\n
            new_tracing_points -- MediaPipe Hands new landmarks for tracing points.\n
            right_hand_detected -- Boolean value to check, if right hand got detected.
            """

            if (right_hand_detected == True):
                self.right_hand_tracing_points_pos = np.roll(self.right_hand_tracing_points_pos, self.one_hand_tracing_points_coordinates_amount) 
                self.right_hand_tracing_points_pos[:self.one_hand_tracing_points_coordinates_amount] = new_tracing_points 
            else :
                self.right_hand_tracing_points_pos = np.roll(self.right_hand_tracing_points_pos, self.one_hand_tracing_points_coordinates_amount) 
                self.right_hand_tracing_points_pos[:self.one_hand_tracing_points_coordinates_amount] = np.zeros(self.one_hand_tracing_points_coordinates_amount) 