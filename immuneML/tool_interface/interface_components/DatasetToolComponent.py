import os
import shutil

from immuneML.tool_interface.interface_components.InterfaceComponent import InterfaceComponent


class DatasetToolComponent(InterfaceComponent):
    def __init__(self, name: str, specs: dict):
        super().__init__(name, specs)

    def run(self):
        # TODO: send request to tool with parameters
        tool_args = self.create_json_params(self.specs)

        # TODO: receive response and
        message_response = self.socket.recv()
        print(f"Message received from the C++ program: {message_response.decode()}")

        # TODO:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        default_dataset_folder = os.path.join(parent_directory, "generated_datasets")

        self.move_file_to_dir(message_response.decode(), default_dataset_folder)

    def move_file_to_dir(self, file_path: str, target_path: str):
        """ Moves a file to a target path
        """
        filename = os.path.basename(file_path)

        # If filename already exists in target path, change its name based on duplicates
        if os.path.exists(target_path):
            i = 1
            while True:
                new_target_path = os.path.join(target_path,
                                               f"{os.path.splitext(filename)[0]}({i}){os.path.splitext(filename)[1]}")
                if not os.path.exists(new_target_path):
                    target_path = new_target_path
                    break
                i += 1

        shutil.move(file_path, target_path)
