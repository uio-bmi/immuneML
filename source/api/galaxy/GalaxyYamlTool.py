from glob import glob
from typing import List

import yaml

from source.api.galaxy.Util import Util
from source.app.ImmuneMLApp import ImmuneMLApp
from source.util.PathBuilder import PathBuilder


class GalaxyYamlTool:

    def __init__(self, yaml_path, output_dir, **kwargs):
        Util.check_parameters(yaml_path, output_dir, kwargs, "Galaxy immuneML Tool")

        self.yaml_path = yaml_path
        self.result_path = output_dir if output_dir[-1] == '/' else f"{output_dir}/"
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
        specs_dict = self.update_yaml_with_collections(specs_dict)
        specs_dict = self.update_result_paths(specs_dict)

        with open(self.yaml_path, "w") as file:
            yaml.dump(specs_dict, file)

    def extract_collection_dataset_paths(self) -> List[str]:
        dataset_paths = list(glob(f"{self.start_path}**/*.iml_dataset", recursive=True))
        return dataset_paths

    def update_result_paths(self, specs: dict) -> dict:
        for key, item in specs["definitions"]["datasets"].items():
            if isinstance(item, dict) and 'params' in item.keys() and isinstance(item["params"], dict):
                item['params']["result_path"] = f"{self.result_path}{key}/"
        return specs

    def update_yaml_with_collections(self, specs: dict) -> dict:
        datasets = specs["definitions"]["datasets"]
        collection_dataset_paths = self.extract_collection_dataset_paths()
        for key, item in datasets.items():
            if isinstance(item, str):
                dataset_file_path = [p for p in collection_dataset_paths if item in p]
                assert len(dataset_file_path) == 1, f"Galaxy immuneML Tool: could not find the dataset collection called {item} " \
                                                    f"specified under key {key}. Please check if the collection was selected properly."
                datasets[key] = {
                    "format": "Pickle",
                    "params": {"path": dataset_file_path[0]}
                }
        specs["definitions"]["datasets"] = datasets
        return specs

    def check_paths(self, specs: dict):
        for key in specs.keys():
            if isinstance(specs[key], str):
                assert "/" not in specs[key] or specs[key] == "./", "Galaxy immuneML Tool: the paths in specification for Galaxy have to " \
                                                                    f"consist only of the filenames as uploaded to Galaxy history " \
                                                                    f"beforehand. The problem occurs for the parameter {key}."
            elif isinstance(specs[key], dict):
                self.check_paths(specs[key])
