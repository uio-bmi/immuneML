import argparse
import os
import sys
import yaml

from source.util.PathBuilder import PathBuilder


def build_specs(args):
    specs = {
        "definitions": {
            "datasets": {
                args.dataset_name: {
                    "format": args.format,
                    "params": {}
                }
            },
        },
        "instructions": {
            "my_dataset_generation_instruction": {
                "type": "DatasetGeneration",
                "datasets": [args.dataset_name],
                "formats": ["Pickle"]
            }
        }
    }

    if args.metadata_file == "":
        specs["definitions"]["datasets"][args.dataset_name]["params"]["path"] = "."
    else:
        specs["definitions"]["datasets"][args.dataset_name]["params"]["metadata_file"] = args.metadata_file

    return specs



def parse_commandline_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for building immuneML defintion YAML for Galaxy Create Dataset tool")
    parser.add_argument("-r", "--format", required=True, help="The format of the repertoire/receptor dataset")
    parser.add_argument("-m", "--metadata_file", default="", help="The metadata file when using a repertoire dataset. When using a receptor dataset, you may supply an empty string.")
    parser.add_argument("-d", "--dataset_name", default="dataset", help="The name of the created dataset.")

    parser.add_argument("-o", "--output_path", required=True, help="Output location for the generated yaml file (directiory).")
    parser.add_argument("-f", "--file_name", default="specs.yaml", help="Output file name for the yaml file. Default name is 'specs.yaml' if not specified.")

    return parser.parse_args(args)


def main(args):
    parsed_args = parse_commandline_arguments(args)
    specs = build_specs(parsed_args)

    PathBuilder.build(parsed_args.output_path)
    output_location = os.path.join(parsed_args.output_path, parsed_args.file_name)

    with open(output_location, "w") as file:
        yaml.dump(specs, file)

    return output_location


if __name__ == "__main__":
    main(sys.argv[1:])