import logging
import os

import yaml


class Util:

    @staticmethod
    def check_parameters(yaml_path, output_dir, kwargs, location):
        assert os.path.isfile(yaml_path), f"{location}: path to the specification is not correct, got {yaml_path}, " \
                                          f"expecting path to a YAML file."

        assert isinstance(output_dir, str) and output_dir != "", f"{location}: output_dir is {output_dir}, " \
                                                                 f"expected path to a folder to store the results."

    @staticmethod
    def check_paths(specs: dict, tool_name: str):
        for key in specs.keys():
            if isinstance(specs[key], str):
                if "/" in specs[key] and specs[key] != "./" and any(name_part in key for name_part in ('path', 'file')):
                    logging.warning(f"{tool_name}: the paths in specification for Galaxy have to consist only of the filenames "
                                    f"as uploaded to Galaxy history beforehand. The problem occurs for the parameter {key}.")
            elif isinstance(specs[key], dict):
                Util.check_paths(specs[key], tool_name)

    @staticmethod
    def update_result_paths(specs: dict, result_path: str, yaml_path: str):
        for key, item in specs["definitions"]["datasets"].items():
            if isinstance(item, dict) and 'params' in item.keys() and isinstance(item["params"], dict):
                item['params']["result_path"] = f"{result_path}{key}/"

        with open(yaml_path, "w") as file:
            yaml.dump(specs, file)
