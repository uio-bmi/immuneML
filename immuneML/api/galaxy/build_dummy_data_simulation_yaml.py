import argparse
import sys
from pathlib import Path

import yaml

from immuneML.util.PathBuilder import PathBuilder

def get_dataset_simulation_specs(args):
    dataset_specs = {"format": f"Random{args.dataset_type.title()}Dataset",
                     "params": {"labels": {f"{args.label_name}": {f"{args.class1_name}": int(args.class_balance) / 100,
                                                                  f"{args.class2_name}": (100 - int(args.class_balance)) / 100}},
                                f"{args.dataset_type}_count": int(args.count)}}

    return {args.dataset_name: dataset_specs}


def build_specs(args):
    specs = {
        "definitions": {
            "datasets": get_dataset_simulation_specs(args),
        },
        "instructions": {
            "my_dataset_generation_instruction": {
                "type": "DatasetExport",
                "datasets": [args.dataset_name],
                "export_formats": ["AIRR"]
            }
        }
    }

    return specs

def parse_commandline_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for building immuneML defintion YAML for Data Simulation tool")

    parser.add_argument("-d", "--dataset_name", default="dataset", help="The name of the dataset to export.")
    parser.add_argument("-t", "--dataset_type", choices=["receptor", "sequence", "repertoire"], required=True, help="The type of the created dataset (receptor/sequence/repertoire).")
    parser.add_argument("-c", "--count", type=int, required=True, help="The number of examples to generate.")
    parser.add_argument("-l", "--label_name", required=True, help="The name of the label to assign to the dataset.")
    parser.add_argument("-x", "--class1_name", required=True, help="The name of the first class to assign to the dataset.")
    parser.add_argument("-y", "--class2_name", required=True, help="The name of the second class to assign to the dataset.")
    parser.add_argument("-b", "--class_balance", type=int, required=True, help="The class balance percentage (in the range 0 - 100).")

    parser.add_argument("-o", "--output_path", required=True, help="Output location for the generated yaml file (directiory).")
    parser.add_argument("-f", "--file_name", default="specs.yaml",
                        help="Output file name for the yaml file. Default name is 'specs.yaml' if not specified.")

    return parser.parse_args(args)

def main(args):
    parsed_args = parse_commandline_arguments(args)
    specs = build_specs(parsed_args)

    PathBuilder.build(parsed_args.output_path)
    output_location = Path(parsed_args.output_path) / parsed_args.file_name

    with output_location.open("w") as file:
        yaml.dump(specs, file)

    return str(output_location)


if __name__ == "__main__":
    main(sys.argv[1:])
