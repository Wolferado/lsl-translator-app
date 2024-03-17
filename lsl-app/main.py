import flet as ft

from components.home_view import Home
from components.recognition_view import Recognition
from components.model_teaching_view import Teaching

def main(page: ft.Page):
    page.title = "LSL Prototype"

    view_placeholder = ft.Column(
        controls=[Home()]
    )

    def change_view(e):
        view_placeholder.controls.clear()

        if navigation_bar.selected_index == 0:
            print("home_screen")
            home_view = Home()
            view_placeholder.controls.append(home_view)
        elif navigation_bar.selected_index == 1:
            print("recognize")
            recognition_view = Recognition()
            view_placeholder.controls.append(recognition_view)
        elif navigation_bar.selected_index == 2:
            print("teach")
            teaching_view = Teaching()
            view_placeholder.controls.append(teaching_view)

        page.update()

    navigation_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationDestination(icon=ft.icons.HOME, selected_icon=ft.icons.HOME_OUTLINED, label="Home Screen"),
            ft.NavigationDestination(icon=ft.icons.CAMERA, selected_icon=ft.icons.CAMERA_OUTLINED, label="Recognize"),
            ft.NavigationDestination(icon=ft.icons.MODEL_TRAINING, selected_icon=ft.icons.MODEL_TRAINING_SHARP, label="Model Teaching")
        ], on_change=change_view
    )

    page.add(view_placeholder, navigation_bar)

ft.app(target=main)