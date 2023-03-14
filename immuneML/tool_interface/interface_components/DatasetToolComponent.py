import os

from immuneML.tool_interface.interface_components.InterfaceComponent import InterfaceComponent


class DatasetToolComponent(InterfaceComponent):
    def __init__(self, tool_path):
        super().__init__()
        self.tool_path = tool_path
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        DEFAULT_DATASET_FOLDER_PATH = os.path.join(parent_directory, "generated_datasets")

    def run(self, ml_specs: dict):
        print("Running dataset tool component")
        self.start_subprocess(self.tool_path)
        self.port = "5556"
        self.open_connection()

        return "path"
