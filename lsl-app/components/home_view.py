import flet as ft

class Home(ft.UserControl):
    def build(self):
        self.container = ft.Column(
            controls=[
                ft.Text(value='Testing', size=30)
            ]
        )

        return self.container
        