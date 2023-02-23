import json
import time
import subprocess
import zmq
from prefect import flow, task, context


class ToolControllerML:
    def __init__(self):
        self.port = '5555'
        self.socket = None

    def start_subprocess(self):
        # Start tool as subprocess
        tool_process = subprocess.Popen(
            ["python", "/Users/oskar/Desktop/ML_tool/tabular_tool.py", "5555"],
            stdin=subprocess.PIPE)

    def open_connection(self):
        context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to tool…")
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:" + self.port)
        print("Connected to tool")

    def close_connection(self):
        self.socket.close()

    def stop_subprocess(self):
        pass

    def run_fit(self, encoded_data):
        # x = {
        #    'fit': 5,
        # }
        #
        # my_data = json.dumps(x)
        # self.socket.send_json(my_data)
        self.socket.send_pyobj(encoded_data)
        result = self.socket.recv_json()
        print("result", result)

        # run fit
        x = {
            'fit': 1,
        }
        self.socket.send_json(json.dumps(x))
        print(self.socket.recv_json())

    def run_predict(self, encoded_data):
        self.socket.send_pyobj(encoded_data)
        result = self.socket.recv_json()
        print("result from run predict", result)

        x = {
            'predict': 1,
        }
        self.socket.send_json(json.dumps(x))
        print(self.socket.recv_json())

    def run_predict_proba(self, encoded_data):
        self.socket.send_pyobj(encoded_data)
        result = self.socket.recv_json()
        print("result", result)

        x = {
            'predict_proba': 1,
        }
        self.socket.send_json(json.dumps(x))
        print(self.socket.recv_json())
