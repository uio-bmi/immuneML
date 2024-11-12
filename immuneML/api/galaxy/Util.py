import logging
from pathlib import Path

import yaml
import os
import glob
import pandas as pd

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
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
    def update_dataset_key(specs: dict, location, new_key="dataset"):
        dataset_keys = list(specs["definitions"]["datasets"].keys())
        assert len(dataset_keys) == 1, f"{location}: one dataset has to be defined under definitions/datasets, got {dataset_keys} instead."

        orig_key = dataset_keys[0]

        if orig_key != "dataset":
            specs["definitions"]["datasets"][new_key] = specs["definitions"]["datasets"][orig_key]
            specs["definitions"]["datasets"].pop(orig_key)

            for instruction_key in specs["instructions"].keys():
                if "dataset" in specs["instructions"][instruction_key]:
                    specs["instructions"][instruction_key]["dataset"] = new_key

                if "datasets" in specs["instructions"][instruction_key]:
                    specs["instructions"][instruction_key]["datasets"] = [new_key]

                if "analyses" in specs["instructions"][instruction_key]:
                    for analysis_key in specs["instructions"][instruction_key]["analyses"].keys():
                        specs["instructions"][instruction_key]["analyses"][analysis_key]["dataset"] = new_key

            logging.info(f"{location}: renamed dataset '{orig_key}' to '{new_key}'.")

    @staticmethod
    def update_result_paths(specs: dict, result_path: Path, yaml_path: Path):
        if 'datasets' in specs['definitions']:
            for key, item in specs["definitions"]["datasets"].items():
                if isinstance(item, dict) and 'params' in item.keys() and isinstance(item["params"], dict):
                    item['params']["result_path"] = str(result_path / key)
                    if item['format'] not in ['RandomRepertoireDataset', 'RandomReceptorDataset', 'RandomSequenceDataset']:
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

    @staticmethod
    def discover_dataset_path(dataset_name="dataset"):
        if os.path.exists(f"{dataset_name}.yaml"):
            dataset_path = f"{dataset_name}.yaml"
        else:
            discovered = glob.glob(f"*{dataset_name}*.yaml")

            if len(discovered) == 1:
                dataset_path = discovered[0]
            else:
                raise FileNotFoundError(f"Unable to locate '{dataset_name}.yaml'")

        return dataset_path


    @staticmethod
    def remove_path_from_filename(file_path):
        return str(Path(file_path).name)

    @staticmethod
    def reformat_galaxy_dataset(galaxy_dataset_path):
        dataset_yaml_file = galaxy_dataset_path / "dataset.yaml"
        assert dataset_yaml_file.is_file(), "Error: generated dataset.yaml not found"

        metadata_file = None

        with (dataset_yaml_file.open("r") as file):
            dataset_params = yaml.load(file, Loader=yaml.SafeLoader)

            if "metadata_file" in dataset_params:
                dataset_params["metadata_file"] = Util.remove_path_from_filename(dataset_params["metadata_file"])
                metadata_file = galaxy_dataset_path / dataset_params["metadata_file"]

            if "filename" in dataset_params:
                dataset_params["filename"] = str(Path(dataset_params["filename"]).name)

        with dataset_yaml_file.open("w") as file:
            yaml.dump(dataset_params, file)

        if metadata_file is not None:
            metadata_content = pd.read_csv(metadata_file, sep=",")
            metadata_content["filename"] = [Util.remove_path_from_filename(filename) for filename in metadata_content["filename"]]
            metadata_content.to_csv(metadata_file, index=None)

    @staticmethod
    def export_galaxy_dataset(dataset, result_path):
        try:
            PathBuilder.build(result_path / 'galaxy_dataset')
            AIRRExporter.export(dataset, result_path / "galaxy_dataset/")
            dataset_file = list(glob.glob(str(result_path / "galaxy_dataset/*.yaml")))[0]
            os.rename(dataset_file, result_path / "galaxy_dataset/dataset.yaml")
            Util.reformat_galaxy_dataset(result_path / "galaxy_dataset/")
        except Exception as e:
            raise RuntimeError(f"Error when exporting Galaxy dataset: {e}.")