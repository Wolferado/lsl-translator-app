import threading
import flet as ft # GUI library
import mediapipe as mp # Library for hand detection
import cv2 # Library for camera manipulations
import base64 # For conversion

mp_hands = mp.solutions.hands # Load the solution from mediapipe library
mp_face_mesh = mp.solutions.face_mesh # Load the solution from mediapipe library
mp_drawing = mp.solutions.drawing_utils # Enabling drawing utilities from MediaPipe library
mp_drawing_styles = mp.solutions.drawing_styles

# Enable a camera for the input (old variant, global, doesn't relaunch, if stopped)
# cap = cv2.VideoCapture(0)

class RecognitionVisualization(ft.UserControl):
    def build(self):
        self.image = ft.Image()
        self.left_hand_visible = False
        self.right_hand_visible = False
        self.left_hand_detected_icon = ft.Icon(name=ft.icons.BACK_HAND_OUTLINED, color=ft.colors.GREY)
        self.face_detected_icon = ft.Icon(name=ft.icons.TAG_FACES_OUTLINED, color=ft.colors.GREY)
        self.right_hand_detected_icon = ft.Icon(name=ft.icons.FRONT_HAND_OUTLINED, color=ft.colors.GREY)
        self.icon_row = ft.Row(
            controls=[self.left_hand_detected_icon, self.face_detected_icon, self.right_hand_detected_icon]
        )
        self.text_field = ft.TextField(
            label="Text Recognition",
            value="Awaiting the sign 👋..."
        )
        
        self.cap = cv2.VideoCapture(0)

        return ft.Column(
            alignment=ft.alignment.center,
            controls=[self.image, self.icon_row, self.text_field]
        )
    
    def did_mount(self): # Controls appeared on the page
        self.th = threading.Thread(target=self.update_timer, args=(), daemon=True)
        self.th.start()

    def will_unmount(self): # Controls disappeared from the page 
        self.th.join()
        self.cap.release() # Fixes endless LED on other pages

    def update_timer(self):
        self.detect_hands_and_face()

    def detect_hands_and_face(self):
        # Create a mask for the hands
        with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.5, max_num_hands=2) as hands, mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=1) as face_mesh:
            while self.cap.isOpened():
                ret, image = self.cap.read()
            
                # if ret == False: return # Stops errors, don't know how crucial, only warns
                 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1) # Flip the stream

                hand_results = hands.process(image) 
                face_results = face_mesh.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.process_and_draw_landmarks(image, hand_results, face_results)

                # Open a window with the app
                ret, image_arr = cv2.imencode(".png", image)
                image_b64 = base64.b64encode(image_arr)
                self.image.src_base64 = image_b64.decode("utf-8")
                self.image.update()
        
    # Method to draw landmarks on face and hands.
    # Parse image (frame) and processed hands and face results
    def process_and_draw_landmarks(self, image, hand_results, face_results):
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

            # For each hand in the frame, draw landamrks
            for hand in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image = image, 
                    landmark_list = hand, 
                    connections = mp_hands.HAND_CONNECTIONS, # Draw landmarks and their connections based on image, hand shown 
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), # Customize landmarks
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2) # Customize connections
                ) 

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
            self.left_hand_visible = False
            self.right_hand_visible = False

        self.icon_row.update()

    