import json
import random
import subprocess

import numpy as np
import zmq

tool_process = None


class ToolControllerML:
    def __init__(self):
        self.tool_path = None
        random_number = random.randint(1, 9)
        self.port = "555" + str(random_number)
        print(type(self.port))
        self.port = "5555"
        self.socket = None
        self.subprocess = None
        self.pid = None

    def start_subprocess(self, tool_path):
        # TODO set port here? Check if port is available.
        self.tool_path = tool_path
        # Start tool as subprocess
        global tool_process
        tool_process = subprocess.Popen(
            ["python", self.tool_path, self.port],
            stdin=subprocess.PIPE)
        self.pid = tool_process.pid

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
        # self.subprocess.terminate()

        # if self.subprocess.poll() is None:
        #    print("subprocess not terminated")
        # else:
        #    print("subprocess stopped")
        # print(self.subprocess.poll())
        global tool_process
        print("stopping tool process", self.pid)
        if tool_process is not None and (self.pid is None or self.pid == self.pid):
            tool_process.kill()
            tool_process = None
        print("tool process stopped")

    def run_fit(self, encoded_data):
        # x = {
        #    'fit': 5,
        # }
        #
        # my_data = json.dumps(x)
        # self.socket.send_json(my_data)
        self.socket.send_pyobj(encoded_data)
        result = json.loads(self.socket.recv_json())
        if result["data_received"] is True:
            print("Tool received data")

        # run fit
        x = {
            'fit': 1,
        }
        self.socket.send_json(json.dumps(x))
        print(self.socket.recv_json())

    def run_predict(self, encoded_data):
        self.socket.send_pyobj(encoded_data)
        result = json.loads(self.socket.recv_json())
        if result["data_received"] is True:
            print("Tool received data")

        x = {
            'predict': 1,
        }
        self.socket.send_json(json.dumps(x))
        final_result = self.socket.recv_json()
        print(json.loads(final_result))
        final_json = json.loads(json.loads(final_result))
        return final_json

    def run_predict_proba(self, encoded_data):
        self.socket.send_pyobj(encoded_data)
        result = json.loads(self.socket.recv_json())
        if result["data_received"] is True:
            print("Tool received data")

        x = {
            'predict_proba': 1,
        }

        self.socket.send_json(json.dumps(x))
        final_result = self.socket.recv_json()
        final_json = json.loads(final_result)
        result_loads = json.loads(final_json)
        final_results = {
            "signal_disease": np.vstack([1 - np.array(result_loads["predictions"]), result_loads["predictions"]]).T}

        return final_results
