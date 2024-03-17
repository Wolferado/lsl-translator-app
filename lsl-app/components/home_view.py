import flet as ft

class Home(ft.UserControl):
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

        return (
            ft.Column(
                controls=[self.app_title, self.app_desc]
            )
        )
        