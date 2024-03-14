import flet as ft

class Teaching(ft.UserControl):
    def __init__(self, page):
        print("Teaching reached")
        page.title = 'Latvian Sign Language Recognition - Teaching Module'

    def build(self):
        return ft.Column(controls=[ft.Text("Testing", size=30, color='black')])