import os

from immuneML.tool_interface.interface_components.InterfaceComponent import InterfaceComponent


class DatasetToolComponent(InterfaceComponent):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    DEFAULT_DATASET_FOLDER_PATH = os.path.join(parent_directory, "generated_datasets")

    @staticmethod
    def run_dataset_tool_component(ml_specs: dict):
        print("Running dataset tool component")
        pass
