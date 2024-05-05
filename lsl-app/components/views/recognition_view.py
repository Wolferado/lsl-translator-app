import flet as ft

from components.visualizations.recognition_visualization import RecognitionVisualization

class RecognitionScreen(ft.UserControl):
    def build(self):
        self.app_title = ft.Container(
            content=ft.Text('Recognition', text_align=ft.TextAlign.CENTER, size=32), 
            padding=10, 
            alignment=ft.alignment.center
        )

        self.start_recognition_btn = ft.ElevatedButton(
            text='Launch "Recognition"',
            icon=ft.icons.FACE_UNLOCK_ROUNDED,
            visible=True,
            on_click=self.start_recognition
        )
        
        self.stop_teach_btn = ft.ElevatedButton(
            text='Close "Recognition"',
            icon=ft.icons.FACE_RETOUCHING_OFF,
            visible=False,
            on_click=self.stop_recognition
        )

        self.recognition_placeholder = ft.Column()

        return ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=5,
            controls=[
                self.app_title,
                self.start_recognition_btn,
                self.stop_teach_btn,
                ft.Divider(),
                self.recognition_placeholder
            ]   
        )

    def start_recognition(self, e):
        self.start_recognition_btn.text = "Loading..."
        self.recognition_placeholder.controls.append(ft.Container(
            content=ft.ProgressRing(),
            alignment=ft.alignment.center)
        )
        self.update()

        self.recognition_placeholder.controls.clear()
        self.start_recognition_btn.visible = False
        self.stop_teach_btn.visible = True

        view = RecognitionVisualization()

        self.recognition_placeholder.controls.append(view)
        self.update()

    def stop_recognition(self, e):
        self.start_recognition_btn.text = 'Launch "Recognition"'
        self.start_recognition_btn.visible = True
        self.stop_teach_btn.visible = False

        self.recognition_placeholder.controls.clear()
        self.update()