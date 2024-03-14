import flet as ft
from components.recognition_view import Recognition
from components.model_teaching_view import Teaching

# TODO: Dive deeper

def main(page: ft.Page):
    def change_view(self, e):
        if navigation_bar.selected_index == 0:
            #page.route = '/recognition_module'
            #page.views.append(Recognition)
            print("recognize")
        elif navigation_bar.selected_index == 1:
            print("teach")
            teaching = Teaching(page)
            place_holder.controls.append(teaching)
            self.update()

    page.title = 'Latvian Sign Language Recognition - Home Screen'
    
    app_title = ft.Container(
        content=ft.Text('Latvian Sign Language Recognition App', text_align=ft.TextAlign.CENTER, size=32), 
        padding=10, 
        alignment=ft.alignment.center
    )
    app_desc = ft.Container(
        content=ft.Text('Translate signs into text!', text_align=ft.TextAlign.CENTER, size=24), 
        padding=10, 
        alignment=ft.alignment.center
    )
    navigation_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationDestination(icon=ft.icons.CAMERA, selected_icon=ft.icons.CAMERA_OUTLINED, label="Recognize"),
            ft.NavigationDestination(icon=ft.icons.MODEL_TRAINING, selected_icon=ft.icons.MODEL_TRAINING_SHARP, label="Model Teaching")
        ],
        on_change=change_view
    )
    place_holder = ft.Column(width=400, height=200)

    page.route = '/'
    page.add(app_title, app_desc, place_holder, navigation_bar)

ft.app(main)
