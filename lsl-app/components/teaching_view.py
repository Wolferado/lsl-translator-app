import flet as ft

from components.teaching_visualization import TeachingVisualization

class TeachingScreen(ft.UserControl):
    def build(self):
        self.learning_directory = None
        self.saving_directory = None
        self.create_folders = True
        self.flip_frame = True

        self.get_learning_directory_dialog = ft.FilePicker(on_result=self.get_learning_directory)
        self.get_saving_directory_dialog = ft.FilePicker(on_result=self.get_saving_directory)

        self.app_title = ft.Container(
            content=ft.Text('Model Teaching', text_align=ft.TextAlign.CENTER, size=32), 
            padding=10, 
            alignment=ft.alignment.center
        )

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

        self.checkbox_create_new_folder = ft.Checkbox(
            label="Create new folder for each file", 
            value=True, 
            tooltip="Set to True, if you wish to create new folders for each video.",
            on_change=self.toggle_create_new_folders_bool
        )

        self.checkbox_flip_video = ft.Checkbox(
            label="Flip images and videos", 
            value=True, 
            tooltip="Set to True, if you wish to record data from both original and flipped videos.",
            on_change=self.toggle_flip_video_bool
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
            alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                self.app_title,
                self.select_learning_directory_btn,
                self.select_saving_directory_btn,
                self.get_learning_directory_dialog,
                self.get_saving_directory_dialog,
                self.checkbox_create_new_folder,
                self.checkbox_flip_video,
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

    def toggle_flip_video_bool(self, e):
        self.flip_frame = not self.flip_frame
        print("Flip: ", self.flip_frame)

    def toggle_create_new_folders_bool(self, e):
        self.create_folders = not self.create_folders
        print("Folder: ", self.create_folders)

    def start_teaching(self, e):
        self.start_teach_btn.visible = False
        self.stop_teach_btn.visible = True
        self.select_learning_directory_btn.disabled = True
        self.select_saving_directory_btn.disabled = True
        self.checkbox_flip_video.disabled = True
        self.checkbox_create_new_folder.disabled = True

        view = TeachingVisualization()
        view.learning_directory = self.learning_directory
        view.saving_directory = self.saving_directory
        view.create_new_folders = self.create_folders
        view.flip_file = self.flip_frame

        self.teaching_placeholder.controls.append(view)
        self.update()

    def stop_teaching(self, e):
        self.start_teach_btn.visible = True
        self.stop_teach_btn.visible = False
        self.select_learning_directory_btn.disabled = False
        self.select_saving_directory_btn.disabled = False
        self.checkbox_flip_video.disabled = False
        self.checkbox_create_new_folder.disabled = False

        self.teaching_placeholder.controls.clear()
        self.update()