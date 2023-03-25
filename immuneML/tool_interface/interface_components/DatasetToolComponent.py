import subprocess

from immuneML.tool_interface.interface_components.InterfaceComponent import InterfaceComponent
import shutil
import json
import zmq
import os


class DatasetToolComponent(InterfaceComponent):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    DEFAULT_DATASET_FOLDER_PATH = os.path.join(parent_directory, "generated_datasets")

    @staticmethod
    def run_dataset_tool_component(specs: dict):
        print("Running dataset tool component")

        tool_args = DatasetToolComponent._create_arguments(specs)
        DatasetToolComponent.start_sub_process(specs)

    @staticmethod
    def move_dataset_to_folder(current_path: str, target_path: str):
        shutil.move(current_path, target_path)

    @staticmethod
    def _create_arguments(specs: dict) -> str:
        """ Creates a json string based on dictionary input. Also adds a port number
        """
        tool_arguments = specs["tool_arguments"]
        json_str = json.dumps(tool_arguments)

        return json_str

    @staticmethod
    def _get_executable_path(specs: dict) -> str:
        if "tool_executable" not in specs or "tool_path" not in specs:
            print("Did not find tool executable. Make sure to add this parameter in YAML spec file")
            return ""

        executable_w_path = specs.get("tool_path") + "/" + specs.get("tool_executable")
        return executable_w_path

    @staticmethod
    def start_sub_process(specs: dict):
        print("Starting subprocess")

        # Using ZeroMQ
        context = zmq.Context()

        # Get executable
        executable_path = DatasetToolComponent._get_executable_path(specs)
        if executable_path == "":
            return

        # Find available port
        port_nr = InterfaceComponent.find_available_port()

        # Set up socket communication with subprocess
        socket = context.socket(zmq.REQ)
        connect_str = "tcp://localhost:" + str(port_nr)  # Add the port number
        socket.connect(connect_str)

        interpreter = InterfaceComponent.get_interpreter(executable_path)
        print(f"Found interpreter: {interpreter}")
        input_data = DatasetToolComponent._create_arguments(specs)
        if not interpreter:
            # No interpreter found. Assuming the file is purely an executable.
            subprocess_args = [executable_path, input_data, str(port_nr)]
        else:
            subprocess_args = [interpreter, executable_path, input_data, str(port_nr)]
        process = subprocess.Popen(subprocess_args,
                                   stdin=subprocess.PIPE,
                                   # stdout=subprocess.PIPE,
                                   cwd=specs.get("tool_path"))  # TODO: specs.get("tool_path") have must error check

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
        InterfaceComponent.move_file_to_dir(message_response.decode(), DatasetToolComponent.DEFAULT_DATASET_FOLDER_PATH)

        print("ENDING DATASET PROCESS")


