import argparse
import sys
from pathlib import Path
import os.path
import logging

from immuneML.api.galaxy.Util import Util
from immuneML.data_model.bnp_util import write_yaml
from immuneML.util.PathBuilder import PathBuilder


def parse_command_line_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for building specification for applying previously trained ML models in Galaxy")

    parser.add_argument("-t", "--trained_model", required=True, help="The trained ML model to apply to the dataset.")

    parser.add_argument("-o", "--output_path", required=True,
                        help="Output location for the generated yaml file (directory).")
    parser.add_argument("-f", "--file_name", default="specs.yaml",
                        help="Output file name for the yaml file. Default name is 'specs.yaml' if not specified.")

    return parser.parse_args(args)


def build_specs(parsed_args):
    specs = {
        "definitions": {
            "datasets": {
                "dataset": {
                    "format": "AIRR",
                    "params": {"dataset_file": Util.discover_dataset_path()}
                }
            },
        },
        "instructions": {
            f"apply_ml_model": {
                "type": "MLApplication",
                "dataset": "dataset",
                "number_of_processes": 8,
                "config_path": parsed_args.trained_model
            }
        }
    }

    return specs

def main(args):
    parsed_args = parse_command_line_arguments(args)
    specs = build_specs(parsed_args)

    if not os.path.isfile(parsed_args.trained_model):
        logging.warning(f"Could not locate trained ML model: {parsed_args.trained_model}")

    PathBuilder.build(parsed_args.output_path)
    output_location = Path(parsed_args.output_path) / parsed_args.file_name

    write_yaml(output_location, specs)

    return str(output_location)


if __name__ == "__main__":
    main(sys.argv[1:])
