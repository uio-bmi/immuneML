import abc
import subprocess
import sys
import time
from abc import ABC

import zmq

tool_process = None


class InterfaceComponent(ABC):
    def __init__(self):
        self.tool_path = None
        self.port = None
        self.socket = None
        self.pid = None
        self.programming_language = None

    @abc.abstractmethod
    def run(self, **args):
        pass

    def start_subprocess(self, tool_path):
        # TODO set port here? Check if port is available.
        self.tool_path = tool_path
        # Start tool as subprocess
        global tool_process
        tool_process = subprocess.Popen(
            ["python", self.tool_path, self.port],
            stdin=subprocess.PIPE)
        self.pid = tool_process.pid

    def stop_subprocess(self):
        global tool_process
        print("stopping tool process", self.pid)
        if tool_process is not None and (self.pid is None or self.pid == self.pid):
            tool_process.kill()
            tool_process = None
        print("tool process stopped")

    def open_connection(self):
        context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to tool…")
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:" + self.port)
        print("Connected to tool")

    def close_connection(self):
        self.socket.close()

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
