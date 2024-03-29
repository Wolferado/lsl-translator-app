import flet as ft

from components.extraction_visualization import ExtractionVisualization

class ExtractionScreen(ft.UserControl):
    def build(self):
        self.data_extraction_directory = None
        self.data_saving_directory = None
        self.create_folders = True
        self.flip_frame = True

        self.get_data_extraction_directory_dialog = ft.FilePicker(on_result=self.get_data_extraction_directory)
        self.get_data_saving_directory_dialog = ft.FilePicker(on_result=self.get_data_saving_directory)

        self.tab_title = ft.Container(
            content=ft.Text('Data Extractor', text_align=ft.TextAlign.CENTER, size=32), 
            padding=10, 
            alignment=ft.alignment.center
        )

        self.select_data_extraction_directory_btn = ft.ElevatedButton(
            text="Select directory to get data from",
            icon=ft.icons.FOLDER,
            on_click=lambda _: self.get_data_extraction_directory_dialog.get_directory_path(dialog_title="Select directory to retrieve the data from")
        )
        
        self.select_data_saving_directory_btn = ft.ElevatedButton(
            text="Select data saving directory",
            icon=ft.icons.FOLDER,
            on_click=lambda _: self.get_data_saving_directory_dialog.get_directory_path(dialog_title="Select directory to where save data")
        )

        self.checkbox_create_new_folder = ft.Checkbox(
            label="Create new folder for each sign", 
            value=True, 
            tooltip="Creates new folder for each subfolder in the selected directory in saving directory. Disable, if you wish to only add data to existing directory.",
            on_change=self.toggle_create_new_folders_bool
        )

        self.checkbox_flip_video = ft.Checkbox(
            label="Flip images and videos", 
            value=True, 
            tooltip="Set to True, if you wish to record data from both original and flipped videos.",
            on_change=self.toggle_flip_video_bool
        )

        self.start_data_extraction_btn = ft.ElevatedButton(
            text="Start data extraction process",
            visible=False,
            on_click=self.start_extraction
        )
        
        self.stop_data_extraction_btn = ft.ElevatedButton(
            text="Stop data extraction process",
            visible=False,
            on_click=self.stop_extraction
        )

        self.data_extraction_placeholder = ft.Column(
            alignment=ft.alignment.center
        )

        return ft.Column(
            alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5,
            controls=[
                self.tab_title,
                self.select_data_extraction_directory_btn,
                self.select_data_saving_directory_btn,
                self.get_data_extraction_directory_dialog,
                self.get_data_saving_directory_dialog,
                self.checkbox_create_new_folder,
                self.checkbox_flip_video,
                self.start_data_extraction_btn,
                self.stop_data_extraction_btn,
                self.data_extraction_placeholder
            ]   
        )
    
    def get_data_extraction_directory(self, e: ft.FilePickerResultEvent):
        self.data_extraction_directory = e.path
        self.show_hide_teach_btn()

        if self.data_extraction_directory:
            self.select_data_extraction_directory_btn.text = "Extraction directory selected: ...\\{}\\{}".format(self.data_extraction_directory.rsplit('\\', 2)[-2], self.data_extraction_directory.rsplit('\\', 2)[-1])
        else:
            self.select_data_extraction_directory_btn.text = "Select directory to get data from"

        self.update()

    def get_data_saving_directory(self, e: ft.FilePickerResultEvent):
        self.data_saving_directory = e.path
        self.show_hide_teach_btn()

        if self.data_saving_directory:
            self.select_data_saving_directory_btn.text = "Saving directory selected: ...\\{}\\{}".format(self.data_saving_directory.rsplit('\\', 2)[-2], self.data_saving_directory.rsplit('\\', 2)[-1])
        else:
            self.select_data_saving_directory_btn.text = "Select data saving directory"

        self.update()

    def show_hide_teach_btn(self):
        if self.data_saving_directory and self.data_extraction_directory:
            self.start_data_extraction_btn.visible = True
        else:
            self.start_data_extraction_btn.visible = False

    def toggle_flip_video_bool(self, e):
        self.flip_frame = not self.flip_frame

    def toggle_create_new_folders_bool(self, e):
        self.create_folders = not self.create_folders

    def start_extraction(self, e):
        self.start_data_extraction_btn.visible = False
        self.stop_data_extraction_btn.visible = True
        self.select_data_extraction_directory_btn.disabled = True
        self.select_data_saving_directory_btn.disabled = True
        self.checkbox_flip_video.disabled = True
        self.checkbox_create_new_folder.disabled = True

        view = ExtractionVisualization()
        view.extraction_directory = self.data_extraction_directory
        view.saving_directory = self.data_saving_directory
        view.create_new_folders = self.create_folders
        view.flip_file = self.flip_frame

        self.data_extraction_placeholder.controls.append(view)
        self.update()

    def stop_extraction(self, e):
        self.start_data_extraction_btn.visible = True
        self.stop_data_extraction_btn.visible = False
        self.select_data_extraction_directory_btn.disabled = False
        self.select_data_saving_directory_btn.disabled = False
        self.checkbox_flip_video.disabled = False
        self.checkbox_create_new_folder.disabled = False

        self.data_extraction_placeholder.controls.clear()
        self.update()