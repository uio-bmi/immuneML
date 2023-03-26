import json
import os

from immuneML.tool_interface.interface_components.InterfaceComponent import InterfaceComponent


class DatasetToolComponent(InterfaceComponent):
    def __init__(self, tool_path):
        super().__init__()
        self.tool_path = tool_path
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        DEFAULT_DATASET_FOLDER_PATH = os.path.join(parent_directory, "generated_datasets")

    def run(self, path: str):
        print("Running dataset tool component")
        self.port = "5556"
        self.start_subprocess(self.tool_path)
        self.open_connection()

        request = {
            'get_dataset': {
                'path': path
            }
        }

        self.socket.send_json(json.dumps(request))
        result = self.socket.recv_json()
        print(json.loads(result))

        self.stop_subprocess()

        return json.loads(result)

    def get_dataset(self, workflow_specification):
        dataset_path = self.run('/Users/oskar/Documents/Skole/Master/immuneml_forked/ML_tool')
        print("Path to dataset created by the tool: ", dataset_path)
        workflow_specification["definitions"]["datasets"]["my_dataset"]["params"]["path"] = dataset_path["path"]
        workflow_specification["definitions"]["datasets"]["my_dataset"]["params"]["metadata_file"] = dataset_path[
            "metadata"]

        return workflow_specification
