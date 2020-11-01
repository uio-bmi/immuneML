import os

import yaml

from source.api.galaxy.Util import Util
from source.app.ImmuneMLApp import ImmuneMLApp
from source.util.PathBuilder import PathBuilder


class GalaxyYamlTool:

    def __init__(self, specification_path, result_path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, "GalaxyYamlTool")

        self.yaml_path = specification_path
        self.result_path = os.path.relpath(result_path) + "/"
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

        Util.check_paths(specs_dict, 'GalaxyYamlTool')
        Util.update_result_paths(specs_dict, self.result_path, self.yaml_path)

