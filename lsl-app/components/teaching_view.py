import flet as ft

from components.teaching_visualization import TeachingVisualization

class TeachingScreen(ft.UserControl):
    def build(self):
        self.learning_directory = None
        self.saving_directory = None

        self.get_learning_directory_dialog = ft.FilePicker(on_result=self.get_learning_directory)
        self.get_saving_directory_dialog = ft.FilePicker(on_result=self.get_saving_directory)

        self.select_learning_directory_btn = ft.ElevatedButton(
            text="Select directory to learn from",
            icon=ft.icons.FOLDER,
            on_click=lambda _: self.get_learning_directory_dialog.get_directory_path(dialog_title="Select directory to retrieve the data from")
        )
        
        self.select_saving_directory_btn = ft.ElevatedButton(
            text="Select data saving directory",
            icon=ft.icons.FOLDER,
            on_click=lambda _: self.get_saving_directory_dialog.get_directory_path(dialog_title="Select directory to where save data")
        )

        self.start_teach_btn = ft.ElevatedButton(
            text="Start model teaching",
            visible=False,
            on_click=self.start_teaching
        )
        
        self.stop_teach_btn = ft.ElevatedButton(
            text="Stop teaching process",
            visible=False,
            on_click=self.stop_teaching
        )

        self.teaching_placeholder = ft.Column(
            alignment=ft.alignment.center
        )

        return ft.Column(
            controls=[
                self.select_learning_directory_btn,
                self.select_saving_directory_btn,
                self.get_learning_directory_dialog,
                self.get_saving_directory_dialog,
                self.start_teach_btn,
                self.stop_teach_btn,
                self.teaching_placeholder
            ]   
        )
    
    def get_learning_directory(self, e: ft.FilePickerResultEvent):
        self.learning_directory = e.path
        self.show_hide_teach_btn()

        if self.learning_directory:
            self.select_learning_directory_btn.text = "Learning directory selected: ...\\{}".format(self.learning_directory.rsplit('\\', 1)[1])
        else:
            self.select_learning_directory_btn.text = "Select directory to learn from"

        self.update()

    def get_saving_directory(self, e: ft.FilePickerResultEvent):
        self.saving_directory = e.path
        self.show_hide_teach_btn()

        if self.saving_directory:
            self.select_saving_directory_btn.text = "Saving directory selected: ...\\{}".format(self.saving_directory.rsplit('\\', 1)[1])
        else:
            self.select_saving_directory_btn.text = "Select data saving directory"

        self.update()

    def show_hide_teach_btn(self):
        if self.saving_directory and self.learning_directory:
            self.start_teach_btn.visible = True
        else:
            self.start_teach_btn.visible = False

    def start_teaching(self, e):
        self.start_teach_btn.visible = False
        self.stop_teach_btn.visible = True
        self.select_learning_directory_btn.disabled = True
        self.select_saving_directory_btn.disabled = True

        view = TeachingVisualization()
        view.learning_directory = self.learning_directory
        view.saving_directory = self.saving_directory

        self.teaching_placeholder.controls.append(view)
        self.update()

    def stop_teaching(self, e):
        self.start_teach_btn.visible = True
        self.stop_teach_btn.visible = False
        self.select_learning_directory_btn.disabled = False
        self.select_saving_directory_btn.disabled = False

        self.teaching_placeholder.controls.clear()
        self.update()