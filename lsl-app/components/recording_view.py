import threading
import flet as ft # GUI library
import mediapipe as mp # Library for hand detection
import cv2 # Library for camera manipulations
import base64 # For conversion
import os

mp_hands = mp.solutions.hands # Load the solution from mediapipe library
mp_face_mesh = mp.solutions.face_mesh # Load the solution from mediapipe library
mp_drawing = mp.solutions.drawing_utils # Enabling drawing utilities from MediaPipe library
mp_drawing_styles = mp.solutions.drawing_styles

class RecordingScreen(ft.UserControl):
    def build(self):
        self.saving_directory = None
        self.recording_started = False
        self.image = ft.Image()
        self.cap = cv2.VideoCapture(0)
        self.out = None
        self.max_frame_amount = 30

        self.get_saving_directory_dialog = ft.FilePicker(on_result=self.get_saving_directory)

        self.app_title = ft.Container(
            content=ft.Text('Data Creator', text_align=ft.TextAlign.CENTER, size=32), 
            padding=10, 
            alignment=ft.alignment.center
        )
        
        self.select_saving_directory_btn = ft.ElevatedButton(
            text="Select folder where to save data",
            icon=ft.icons.FOLDER,
            on_click=lambda _: self.get_saving_directory_dialog.get_directory_path(dialog_title="Select directory to where save data")
        )

        self.take_picture_btn = ft.ElevatedButton(
            text="Take a picture",
            icon=ft.icons.CAMERA_ALT_OUTLINED,
            disabled=True,
            visible=True,
            on_click=self.take_picture
        )
        
        self.start_recording_btn = ft.ElevatedButton(
            text="Start video recording",
            icon=ft.icons.VIDEO_CAMERA_BACK_OUTLINED,
            disabled=True,
            visible=True,
            on_click=self.start_recording
        )

        self.stop_recording_btn = ft.ElevatedButton(
            text="Stop video recording",
            color="#ff0000",
            icon=ft.icons.STOP_CIRCLE_OUTLINED,
            icon_color="#ff0000",
            visible=False,
            on_click=self.stop_recording
        )

        self.camera_placeholder = ft.Column(
            controls=[
                self.image
            ]
        )

        return ft.Column(
            alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                self.app_title,
                self.select_saving_directory_btn,
                self.get_saving_directory_dialog,
                self.take_picture_btn,
                self.start_recording_btn,
                self.stop_recording_btn,
                self.camera_placeholder
            ]   
        )
    
    def did_mount(self):
        self.th = threading.Thread(target=self.update_timer, args=(), daemon=True)
        self.th.start()

    def will_unmount(self): 
        self.th.join()
        self.cap.release() 

    def update_timer(self):
        self.detect_hands_and_face()

    def get_saving_directory(self, e: ft.FilePickerResultEvent):
        self.saving_directory = e.path
        self.enable_disable_control_btn()

        if self.saving_directory:
            self.select_saving_directory_btn.text = "Saving directory selected: ...\\{}\\{}".format(self.saving_directory.rsplit('\\', 2)[-2], self.saving_directory.rsplit('\\', 2)[-1])
        else:
            self.select_saving_directory_btn.text = "Select data saving directory"

        self.update()

    def enable_disable_control_btn(self):
        if self.saving_directory:
            self.take_picture_btn.disabled = False
            self.start_recording_btn.disabled = False
        else:
            self.take_picture_btn.disabled = True
            self.start_recording_btn.disabled = True


    def take_picture(self, e):
        print("Picture taken")

        cv2.imwrite("{}/{}.jpg".format(self.saving_directory, self.count_files()), self.original_image)

        self.update()

    def start_recording(self, e):
        self.start_recording_btn.visible = False
        self.stop_recording_btn.visible = True

        self.out = cv2.VideoWriter("{}/{}.mp4".format(self.saving_directory, self.count_files()), -1, 20.0, (640, 480))
        self.recording_started = True

        self.update()

    def stop_recording(self, e):
        self.start_recording_btn.visible = True
        self.stop_recording_btn.visible = False

        self.recording_started = False
        self.out.release()

        self.update()

    def stop_recording_lambda(self):
        self.start_recording_btn.visible = True
        self.stop_recording_btn.visible = False

        self.recording_started = False
        self.out.release()

        self.update()

    def detect_hands_and_face(self):
        with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.5, max_num_hands=2) as hands, mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=1) as face_mesh:
            while self.cap.isOpened():
                ret, image = self.cap.read()

                image = cv2.flip(image, 1) # Flip the stream
                self.original_image = image

                if(self.recording_started == True and self.max_frame_amount >= 0):
                    self.out.write(self.original_image)
                    self.max_frame_amount -= 1
                elif (self.recording_started == True and self.max_frame_amount < 0):
                    self.stop_recording_lambda()
                    self.max_frame_amount = 30

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False # Disables any modifications of the 2D array
                hand_results = hands.process(image) 
                face_results = face_mesh.process(image)
                image.flags.writeable = True # Allows any modifications of the 2D array
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.draw_landmarks(image, hand_results, face_results)

                # Open a window with the app
                ret, image_arr = cv2.imencode(".png", image)
                image_b64 = base64.b64encode(image_arr)
                self.image.src_base64 = image_b64.decode("utf-8")
                self.image.update()

    # Method to draw landmarks on face and hands.
    # Parse image (frame) and processed hands and face results
    def draw_landmarks(self, image, hand_results, face_results):
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
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2) # Customize connections
                ) 

    def count_files(self) -> int:
        total_file_amount = 0

        for file in os.scandir(self.saving_directory):
            if file.is_file():
                total_file_amount += 1

        return total_file_amount