import os

import yaml

from source.app.ImmuneMLApp import ImmuneMLApp
from source.util.PathBuilder import PathBuilder


class GalaxyYamlTool:

    def __init__(self, yaml_path, output_dir, **kwargs):
        assert os.path.isfile(yaml_path), f"Galaxy immuneML tool: path to the analysis specification is not correct, got {yaml_path}, " \
                                          f"expecting path to YAML file."

        assert isinstance(output_dir, str) and output_dir != "", f"Galaxy immuneML tool: output_dir is {output_dir}, " \
                                                                 f"expected path to a folder to store the results."

        assert "inputs" not in kwargs or all(os.path.dirname(kwargs["inputs"][0]) == os.path.dirname(elem) for elem in kwargs["inputs"]), \
            f"Galaxy immuneML tool: not all repertoire files are under the same directory. " \
            f"Instead, they are in {str(os.path.dirname(elem) for elem in kwargs['inputs'])[1:-1]}."

        self.yaml_path = yaml_path
        self.result_path = output_dir
        self.metadata_file = kwargs["metadata_file"] if "metadata_file" in kwargs else None
        self.files_path = f"{os.path.dirname(kwargs['inputs'][0])}/" if "inputs" in kwargs else None

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

            dataset_keys = specs_dict["definitions"]["datasets"].keys()
            assert len(dataset_keys) == 1, "Galaxy immunneML tool: when using immuneML from Galaxy, " \
                                           "multiple datasets are not yet supported."

            specs_dict["definitions"]["datasets"][dataset_keys[0]]["params"]["metadata_file"] = self.metadata_file
            specs_dict["definitions"]["datasets"][dataset_keys[0]]["params"]["path"] = self.files_path
            specs_dict["output"] = {"format": "HTML"}

            with open(self.yaml_path, "w") as file:
                yaml.dump(specs_dict, file)
