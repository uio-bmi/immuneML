import logging
from pathlib import Path

import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class Util:

    @staticmethod
    def check_parameters(yaml_path: Path, output_dir: Path, kwargs, location):
        assert isinstance(yaml_path, Path), f"{location}: yaml_path is {output_dir}, expected Path object."
        assert isinstance(output_dir, Path), f"{location}: output_dir is {output_dir}, expected Path object pointing to a folder to store the results."

        assert yaml_path.is_file(), f"{location}: path to the specification is not correct, got {yaml_path}, expecting path to a YAML file."


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
                item['params']["result_path"] = str(result_path / key)
                if item['format'] not in ['Pickle', 'RandomRepertoireDataset', 'RandomReceptorDataset']:
                    item['params']['path'] = str(yaml_path.parent)

        with yaml_path.open("w") as file:
            yaml.dump(specs, file)

    @staticmethod
    def check_instruction_type(specs: dict, tool_name, expected_instruction) -> str:
        ParameterValidator.assert_keys_present(list(specs.keys()), ['definitions', 'instructions'], tool_name, "YAML specification")
        assert len(list(specs['instructions'].keys())) == 1, f"{tool_name}: multiple instructions were given " \
                                                             f"({str(list(specs['instructions'].keys()))[1:-1]}), but only one instruction of type " \
                                                             f"{expected_instruction} should be specified."
        instruction_name = list(specs['instructions'].keys())[0]
        instruction_type = specs['instructions'][instruction_name]['type']
        assert instruction_type == expected_instruction, \
            f"{tool_name}: instruction type has to be '{expected_instruction}', got {instruction_type} instead."

        return instruction_name

    @staticmethod
    def check_export_format(specs: dict, tool_name: str, instruction_name: str):
        ParameterValidator.assert_keys_present(list(specs['instructions'][instruction_name].keys()), ["export_formats"], tool_name,
                                               f"{instruction_name}/export_formats")
        ParameterValidator.assert_type_and_value(specs['instructions'][instruction_name]["export_formats"], list, tool_name,
                                                 f"{instruction_name}/export_formats")

        assert len(specs['instructions'][instruction_name]["export_formats"]) == 1, \
            f"{tool_name}: only one format can be specified under export_formats parameter under " \
            f"{instruction_name}/export_formats, got {specs['instructions'][instruction_name]['export_formats']} instead."

        return specs['instructions'][instruction_name]["export_formats"][0]

    @staticmethod
    def run_tool(yaml_path, result_path):
        PathBuilder.build(result_path)
        app = ImmuneMLApp(yaml_path, result_path)
        app.run()
