import flet as ft

class Teaching(ft.UserControl):
    def build(self):
        print("Teaching reached")
        title = 'Latvian Sign Language Recognition - Teaching Module'

        return ft.Column(
            controls=[
                ft.Text("Testing", size=30)
            ]   
        )