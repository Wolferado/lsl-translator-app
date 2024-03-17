import flet as ft

from components.home_view import Home
from components.recognition_view import Recognition
from components.model_teaching_view import Teaching

class LSLApp(ft.UserControl):
    def build(self):
        self.app_title = ft.Container(
            content=ft.Text('Latvian Sign Language Recognition App', text_align=ft.TextAlign.CENTER, size=32), 
            padding=10, 
            alignment=ft.alignment.center
        )
        self.app_desc = ft.Container(
            content=ft.Text('Translate signs into text!', text_align=ft.TextAlign.CENTER, size=24), 
            padding=10, 
            alignment=ft.alignment.center
        )
        self.view_placeholder = ft.Column(
            controls=[Home()]
        )
        self.navigation_bar = ft.NavigationBar(
        destinations=[
                ft.NavigationDestination(icon=ft.icons.HOME, selected_icon=ft.icons.HOME_OUTLINED, label="Home Screen"),
                ft.NavigationDestination(icon=ft.icons.CAMERA, selected_icon=ft.icons.CAMERA_OUTLINED, label="Recognize"),
                ft.NavigationDestination(icon=ft.icons.MODEL_TRAINING, selected_icon=ft.icons.MODEL_TRAINING_SHARP, label="Model Teaching")
            ],
            on_change=self.change_view
        )

        return (
            ft.Column(
                controls=[self.app_title, self.app_desc, self.view_placeholder, self.navigation_bar]
            )
        )

    def change_view(self, e):
        self.view_placeholder.controls.clear()

        if self.navigation_bar.selected_index == 0:
            print("home_screen")
            home_view = Home()
            self.view_placeholder.controls.append(home_view)
        elif self.navigation_bar.selected_index == 1:
            print("recognize")
            recognition_view = Recognition()
            self.view_placeholder.controls.append(recognition_view)
        elif self.navigation_bar.selected_index == 2:
            print("teach")
            teaching_view = Teaching()
            self.view_placeholder.controls.append(teaching_view)


        self.update()
        
        