import flet as ft
from components.app import LSLApp

def main(page: ft.Page):
    page.title = "LSL Prototype"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.update()

    # create application instance
    lsl_app = LSLApp()

    # add application's root control to the page
    page.add(lsl_app)

ft.app(target=main)