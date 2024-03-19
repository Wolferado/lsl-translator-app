import flet as ft

from components.recognition_visualization import RecognitionVisualization

class RecognitionScreen(ft.UserControl):
    def build(self):
        self.start_recognition_btn = ft.ElevatedButton(
            text="Start Recognition",
            icon=ft.icons.FACE_UNLOCK_ROUNDED,
            visible=True,
            on_click=self.start_recognition
        )
        
        self.stop_teach_btn = ft.ElevatedButton(
            bgcolor="770409",
            text="Stop Recognition",
            color="#ff0000",
            icon=ft.icons.FACE_ROUNDED,
            icon_color="#ff0000",
            visible=False,
            on_click=self.stop_recognition
        )

        self.recognition_placeholder = ft.Column()

        return ft.Column(
            controls=[
                self.start_recognition_btn,
                self.stop_teach_btn,
                self.recognition_placeholder
            ]   
        )

    def start_recognition(self, e):
        self.start_recognition_btn.text = "Loading..."
        self.update()

        self.start_recognition_btn.visible = False
        self.stop_teach_btn.visible = True

        view = RecognitionVisualization()

        self.recognition_placeholder.controls.append(view)
        self.update()

    def stop_recognition(self, e):
        self.start_recognition_btn.text = "Start Recognition"
        self.start_recognition_btn.visible = True
        self.stop_teach_btn.visible = False

        self.recognition_placeholder.controls.clear()
        self.update()