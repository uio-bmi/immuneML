import logging
import os

import yaml

from source.api.galaxy.Util import Util
from source.app.ImmuneMLApp import ImmuneMLApp
from source.util.PathBuilder import PathBuilder


class GalaxyYamlTool:

    def __init__(self, yaml_path, output_dir, **kwargs):
        Util.check_parameters(yaml_path, output_dir, kwargs, "Galaxy immuneML Tool")

        self.yaml_path = yaml_path
        self.result_path = os.path.relpath(output_dir) + "/"
        self.start_path = "./"

    def run(self):
        PathBuilder.build(self.result_path)
        self.update_specs()

        app = ImmuneMLApp(self.yaml_path, self.result_path)
        output_file_path = app.run()

        return output_file_path

    def update_specs(self):
        with open(self.yaml_path, "r") as file:
            specs_dict = yaml.safe_load(file)

        self.check_paths(specs_dict)
        specs_dict = self.update_result_paths(specs_dict)

        with open(self.yaml_path, "w") as file:
            yaml.dump(specs_dict, file)

    def update_result_paths(self, specs: dict) -> dict:
        for key, item in specs["definitions"]["datasets"].items():
            if isinstance(item, dict) and 'params' in item.keys() and isinstance(item["params"], dict):
                item['params']["result_path"] = f"{self.result_path}{key}/"
        return specs

    def check_paths(self, specs: dict):
        for key in specs.keys():
            if isinstance(specs[key], str):
                if "/" in specs[key] and specs[key] != "./":
                    logging.warning("Galaxy immuneML Tool: the paths in specification for Galaxy have to consist only of the filenames "
                                    f"as uploaded to Galaxy history beforehand. The problem occurs for the parameter {key}.")
            elif isinstance(specs[key], dict):
                self.check_paths(specs[key])
