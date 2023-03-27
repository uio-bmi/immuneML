import json
import os
import shutil
import subprocess
import zmq

from immuneML.tool_interface.interface_components.InterfaceComponent import InterfaceComponent


class DatasetToolComponent(InterfaceComponent):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    DEFAULT_DATASET_FOLDER_PATH = os.path.join(parent_directory, "generated_datasets")

    def __init__(self, name: str, specs: dict):
        super().__init__(name, specs)
        # super().__init__()
        # self.name = name
        # self.tool_path = specs['path']
        # self.specs = specs
        # self.interpreter = super().get_interpreter(self.tool_path)

    @staticmethod
    def run_dataset_tool_component(specs: dict):
        print("Running dataset tool component")

        tool_args = super().create_json_params(specs)
        DatasetToolComponent.start_sub_process(specs)

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

    def start_sub_process(self, specs: dict):
        print("Starting subprocess")

        # Using ZeroMQ
        context = zmq.Context()

        # Get executable
        executable_path = specs.get('path')

        # Find available port
        port_nr = InterfaceComponent.find_available_port()

        # Set up socket communication with subprocess
        socket = context.socket(zmq.REQ)
        connect_str = "tcp://localhost:" + str(port_nr)  # Add the port number
        socket.connect(connect_str)

        interpreter = super().get_interpreter(executable_path)
        print(f"Found interpreter: {interpreter}")
        input_data = super().create_json_params(specs)
        if not interpreter:
            # No interpreter found. Assuming the file is purely an executable.
            subprocess_args = [executable_path, input_data, str(port_nr)]
        else:
            subprocess_args = [interpreter, executable_path, input_data, str(port_nr)]
        process = subprocess.Popen(subprocess_args,
                                   stdin=subprocess.PIPE,
                                   # stdout=subprocess.PIPE,
                                   cwd=os.path.dirname(specs.get("path")))

        # Send the input data to the subprocess
        # socket.send_string(input_data)

        # Wait for the response. This should be the data used further
        # We need to specify the requirements of the response. Could be as easy as the path to the new dataset
        message_response = socket.recv()
        print(f"Message received from the C++ program: {message_response.decode()}")

        # Close the socket and context
        socket.close()
        context.term()

        # Terminate subprocess
        process.terminate()  # TODO: not sure if this is how it is done when we just want the subprocess to end itself

        # TODO: should probably have some error check here
        # Moves the dataset provided through path to a folder decided by immuneML
        self.move_file_to_dir(message_response.decode(), DatasetToolComponent.DEFAULT_DATASET_FOLDER_PATH)

        print("ENDING DATASET PROCESS")
