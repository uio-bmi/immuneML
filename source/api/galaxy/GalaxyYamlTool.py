import os

import yaml

from source.api.galaxy.Util import Util
from source.app.ImmuneMLApp import ImmuneMLApp
from source.util.PathBuilder import PathBuilder


class GalaxyYamlTool:

    def __init__(self, yaml_path, output_dir, **kwargs):
        Util.check_parameters(yaml_path, output_dir, kwargs, "Galaxy immuneML tool")

        inputs = kwargs["inputs"].split(',') if "inputs" in kwargs else None

        self.yaml_path = yaml_path
        self.result_path = output_dir
        self.metadata_file = kwargs["metadata"] if "metadata" in kwargs else None
        self.files_path = f"{os.path.dirname(inputs[0])}/" if "inputs" in kwargs else None

    def run(self):
        PathBuilder.build(self.result_path)
        self.update_specs()

        app = ImmuneMLApp(self.yaml_path, self.result_path)
        output_file_path = app.run()

        return output_file_path

    def update_specs(self):
        if self.metadata_file is not None:
            with open(self.yaml_path, "r") as file:
                specs_dict = yaml.safe_load(file)

            dataset_keys = list(specs_dict["definitions"]["datasets"].keys())
            assert len(dataset_keys) == 1, "Galaxy immunneML tool: when using immuneML from Galaxy, " \
                                           "multiple datasets are not yet supported."

            specs_dict["definitions"]["datasets"][dataset_keys[0]]["params"]["metadata_file"] = self.metadata_file
            specs_dict["definitions"]["datasets"][dataset_keys[0]]["params"]["path"] = self.files_path
            specs_dict["definitions"]["datasets"][dataset_keys[0]]["params"]["result_path"] = self.result_path + "imported_data/"

            specs_dict["output"] = {"format": "HTML"}

            with open(self.yaml_path, "w") as file:
                yaml.dump(specs_dict, file)
