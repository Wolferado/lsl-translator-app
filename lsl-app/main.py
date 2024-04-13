import flet as ft

from components.views.home_view import HomeScreen
from components.views.recognition_view import RecognitionScreen
from components.views.extraction_view import ExtractionScreen
from components.views.recording_view import RecordingScreen

def main(page: ft.Page):
    page.title = "LSL Prototype"
    page.window_width = 500
    page.window_height = 750
    page.window_resizable = False
    page.window_maximizable = False
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary='#9f2828',
            on_primary='#ffffff',
            primary_container='#ffc5c5',
            on_primary_container='#340707',
            secondary='#6f5555',
            on_secondary='#ffffff',
            secondary_container='#f5dada',
            on_secondary_container='#2a1313',
            tertiary='#766a57',
            on_tertiary='#ffffff',
            tertiary_container='#fdf1db'
        )
    )

    view_placeholder = ft.Column(
        controls=[HomeScreen()]
    )

    def change_view(e):
        view_placeholder.controls.clear()

        if navigation_bar.selected_index == 0:
            home_view = HomeScreen()
            view_placeholder.controls.append(home_view)
        elif navigation_bar.selected_index == 1:
            recognition_view = RecognitionScreen()
            view_placeholder.controls.append(recognition_view)
        elif navigation_bar.selected_index == 2:
            data_extractor_view = ExtractionScreen()
            view_placeholder.controls.append(data_extractor_view)
        elif navigation_bar.selected_index == 3:
            data_creator_view = RecordingScreen()
            view_placeholder.controls.append(data_creator_view)

        page.update()

    navigation_bar = ft.NavigationBar(
        bgcolor='#D49B9B',
        indicator_shape=ft.RoundedRectangleBorder(radius=20),
        destinations=[
            ft.NavigationDestination(icon=ft.icons.HOME, selected_icon=ft.icons.HOME_OUTLINED, label="Home Screen"),
            ft.NavigationDestination(icon=ft.icons.CAMERA, selected_icon=ft.icons.CAMERA_OUTLINED, label="Recognize"),
            ft.NavigationDestination(icon=ft.icons.DATA_ARRAY, selected_icon=ft.icons.DATA_ARRAY_OUTLINED, label="Data Extractor"),
            ft.NavigationDestination(icon=ft.icons.ADD_PHOTO_ALTERNATE, selected_icon=ft.icons.ADD_PHOTO_ALTERNATE_OUTLINED, label="Data Creator")
        ], on_change=change_view
    )

    page.add(view_placeholder, navigation_bar)

if __name__ == "__main__":
    ft.app(target=main)