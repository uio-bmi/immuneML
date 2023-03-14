import abc
from abc import ABC
import subprocess
import socket
import shutil
import json
import time
import sys
import os


class InterfaceComponent(metaclass=abc.ABCMeta):

    interpreters = {
        ".py": "python",
        ".class": "java"
    }

    @classmethod
    def _get_interpreters(cls):
        """ Returns the dictionary of interpreters. Not accessible by child classes
        """
        return cls.interpreters

    @staticmethod
    def get_interpreter(executable: str):
        """ Returns the correct interpreter for executable input
        """
        interpreters = InterfaceComponent._get_interpreters()
        file_extension = os.path.splitext(executable)[1]
        if file_extension not in interpreters:
            print(f"Interpreter not found for executable: {executable}")
            return None

        interpreter = interpreters.get(file_extension)

        return interpreter

    @staticmethod
    def find_available_port(start_port=5000, end_port=8000):
        """ Finds an available port on the computer to send to subprocess
        """
        for port in range(start_port, end_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("", port))
                    return port
                except OSError:
                    pass

        return None

    @staticmethod
    def move_file_to_dir(file_path: str, target_path: str):
        # TODO: does not handle the case where a file with the same name already exists
        shutil.move(file_path, target_path)

    @staticmethod
    def produce_JSON_object(**input_data):
        """Produces a JSON object from input data
        Returns JSON object on success, or None if error
        """

        try:
            json_bytes = json.dumps(input_data)
        except Exception as e:
            print(f"Error: {e}")
            return None

        return json_bytes

    @staticmethod
    def execution_animation(process: subprocess):
        """Function creates an animation to give user feedback while process in running
        """

        num_dots = 0

        while process.poll() is None:
            sys.stdout.write('\rRunning subprocess{}   '.format('.' * num_dots))
            sys.stdout.flush()
            num_dots = (num_dots + 1) % 4
            time.sleep(0.5)

        sys.stdout.flush()
        sys.stdout.write("\rSubprocess finished")

    @staticmethod
    def show_process_output(ml_specs: dict):
        """ Returns true or false for showing process output based on YAML spec file
        """

        show_output = False
        value = ml_specs.get("show_process_output")

        if value is not None:
            if value.lower() == "true":
                show_output = True
            elif value.lower() != "false":
                print(f"Show_process_output must have parameter 'true' or 'false'. Parameter given: {value}")
            else:
                print("""Process output not showing. Include 'Show_process_output: true' in the YAML spec file to 
                see process output""")

        return show_output
