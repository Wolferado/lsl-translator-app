import flet as ft

class HomeScreen(ft.UserControl):
    def build(self):
        self.app_title = ft.Container(
            content=ft.Text('Latvian Sign Language Recognition App', text_align=ft.TextAlign.CENTER, size=32, weight=ft.FontWeight.W_600), 
            padding=10, 
            alignment=ft.alignment.center
        )
        self.app_desc = ft.Container(
            content=ft.Text('Translate signs into text!', text_align=ft.TextAlign.CENTER, size=24, weight=ft.FontWeight.W_500, italic=True), 
            padding=10, 
            alignment=ft.alignment.center
        )
        self.app_hints = ft.Container(
            content=ft.Text('- Open "Recognize" tab to translate sings in the real time! \n\n\n- Use "Data Extractor" tab to get data from images and video files! \n\n\n- Open "Data Creator" tab to create your own datasets!', size=20, weight=ft.FontWeight.W_400),
            padding=20
        )
        self.logo = ft.Container(
            content=ft.Image(src=f"../assets/icons/fav_icon.png", width=75, height=75),
            padding=0,
            alignment=ft.alignment.center
        )

        return (
            ft.Column(
                controls=[self.logo, self.app_title, self.app_desc, self.app_hints]
            )
        )
        