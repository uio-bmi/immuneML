import argparse
import sys
from pathlib import Path

import yaml

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.util.PathBuilder import PathBuilder


def build_metadata_column_mapping(columns_str):
    colnames = columns_str.split(",")
    colnames = [label.strip().strip("'\"") for label in colnames]

    return {colname: colname for colname in colnames if colname != ""}


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
                "type": "DatasetExport",
                "datasets": [args.dataset_name],
                "export_formats": ["Pickle"]
            }
        }
    }

    specs["definitions"]["datasets"][args.dataset_name]["params"]["region_type"] = RegionType.IMGT_CDR3.name
    specs["definitions"]["datasets"][args.dataset_name]["params"]["result_path"] = "./"
    specs["definitions"]["datasets"][args.dataset_name]["params"]["path"] = "./"

    if args.is_repertoire == "True":
        specs["definitions"]["datasets"][args.dataset_name]["params"]["is_repertoire"] = True
        specs["definitions"]["datasets"][args.dataset_name]["params"]["metadata_file"] = args.metadata_file
    else:
        specs["definitions"]["datasets"][args.dataset_name]["params"]["is_repertoire"] = False

        paired = True if args.paired == "True" else False

        specs["definitions"]["datasets"][args.dataset_name]["params"]["paired"] = paired
        if paired:
            specs["definitions"]["datasets"][args.dataset_name]["params"]["receptor_chains"] = args.receptor_chains

        if args.metadata_columns != "":
            specs["definitions"]["datasets"][args.dataset_name]["params"]["metadata_column_mapping"] = build_metadata_column_mapping(args.metadata_columns)

    return specs


def parse_commandline_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for building immuneML defintion YAML for Galaxy Create Dataset tool")
    parser.add_argument("-r", "--format", required=True, help="The format of the repertoire/receptor dataset")
    parser.add_argument("-m", "--metadata_file", default="",
                        help="The metadata file when using a repertoire dataset. When using a receptor dataset, you may supply an empty string.")
    parser.add_argument("-i", "--is_repertoire", choices=["True", "False"], required=True, help="Whether to import a RepertoireDataset")
    parser.add_argument("-p", "--paired", choices=["True", "False"], default="False",
                        help="When the data is not repertoire data (metadata file = ''), this specifies whether the data is paired (ReceptorDataset) or unpaired (SequenceDataset)")
    parser.add_argument("-c", "--receptor_chains", choices=["TRA_TRB", "TRG_TRD", "IGH_IGL", "IGH_IGK"], default="TRA_TRB",
                        help="When the data is a ReceptorDataset, this specifies the type of receptor chains that are used.")
    parser.add_argument("-a", "--metadata_columns", default="", help="The name of metadata columns of a Sequence- or ReceptorDataset.")
    parser.add_argument("-d", "--dataset_name", default="dataset", help="The name of the created dataset.")

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
