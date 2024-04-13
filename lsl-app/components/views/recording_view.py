import flet as ft

from components.visualizations.recording_visualization import RecordingVisualization

class RecordingScreen(ft.UserControl):
    def build(self):
        self.app_title = ft.Container(
            content=ft.Text('Data Creator', text_align=ft.TextAlign.CENTER, size=32), 
            padding=10, 
            alignment=ft.alignment.center
        )

        self.start_recognition_btn = ft.ElevatedButton(
            text='Launch "Data Creator"',
            icon=ft.icons.APPS_ROUNDED,
            visible=True,
            on_click=self.launch_data_creator
        )
        
        self.stop_teach_btn = ft.ElevatedButton(
            text='Close "Data Creator"',
            icon=ft.icons.APPS_ROUNDED,
            visible=False,
            on_click=self.close_data_creator
        )

        self.data_creator_placeholder = ft.Column()

        return ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=5,
            controls=[
                self.app_title,
                self.start_recognition_btn,
                self.stop_teach_btn,
                ft.Divider(),
                self.data_creator_placeholder
            ]   
        )

    def launch_data_creator(self, e):
        self.start_recognition_btn.text = "Loading..."
        self.data_creator_placeholder.controls.append(ft.Container(
            content=ft.ProgressRing(),
            alignment=ft.alignment.center)
        )
        self.update()

        self.data_creator_placeholder.controls.clear()
        self.start_recognition_btn.visible = False
        self.stop_teach_btn.visible = True

        view = RecordingVisualization()

        self.data_creator_placeholder.controls.append(view)
        self.update()

    def close_data_creator(self, e):
        self.start_recognition_btn.text = 'Launch "Data Creator"'
        self.start_recognition_btn.visible = True
        self.stop_teach_btn.visible = False

        self.data_creator_placeholder.controls.clear()
        self.update()