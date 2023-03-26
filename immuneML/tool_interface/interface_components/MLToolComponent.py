import json

import numpy as np

from immuneML.tool_interface.interface_components.InterfaceComponent import InterfaceComponent


class MLToolComponent(InterfaceComponent):
    def __init__(self, port):
        super().__init__()
        self.port = port

    def run_fit(self, encoded_data):
        self.socket.send_pyobj(encoded_data)
        result = json.loads(self.socket.recv_json())
        if result["data_received"] is True:
            print("Tool received data")

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
