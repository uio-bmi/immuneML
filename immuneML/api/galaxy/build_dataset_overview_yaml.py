import argparse
import sys
from pathlib import Path

import yaml
import os

from immuneML.api.galaxy.build_dataset_yaml import get_dataset_specs
from immuneML.util.PathBuilder import PathBuilder


def add_report_with_label(specs, args, report_name, report_key):
    if args.label_name != "":
        specs["definitions"]["reports"][f"{report_key}_report"] = {report_name: {"label": args.label_name}}
    else:
        specs["definitions"]["reports"][f"{report_key}_report"] = report_name

    specs["instructions"]["my_dataset_generation_instruction"]["analyses"][f"{report_key}_analysis"] = {
        "dataset": args.dataset_name, "report": f"{report_key}_report"}

def build_specs(args):
    specs = {
        "definitions": {
            "datasets": dict(),
            "reports": dict()
        },
        "instructions": {
            "my_dataset_generation_instruction": {
                "type": "ExploratoryAnalysis",
                "analyses": dict()
            }
        }
    }

    if args.existing_dataset == "True":
        assert os.path.exists(f"{args.dataset_name}.yaml"), f"no '{args.dataset_name}.yaml' file was present in the current working directory"
        specs["definitions"]["datasets"][args.dataset_name] = {"format": "ImmuneML",
                                                       "params": {"path": f"{args.dataset_name}.yaml"}}
    else:
        specs["definitions"]["datasets"] = get_dataset_specs(args)

    if args.sequence_length_report == "True":
        specs["definitions"]["reports"]["sequence_length_report"] = "SequenceLengthDistribution"
        specs["instructions"]["my_dataset_generation_instruction"]["analyses"]["sequence_length_analysis"] = {
            "dataset": args.dataset_name, "report": "sequence_length_report"}

    if args.vj_gene_report == "True":
        add_report_with_label(specs, args, report_name="VJGeneDistribution", report_key="vj_gene")

    if args.amino_acid_report == "True":
        add_report_with_label(specs, args, report_name="AminoAcidFrequencyDistribution", report_key="amino_acid")

    if len(specs["instructions"]["my_dataset_generation_instruction"]["analyses"]) == 0:
        specs["instructions"]["my_dataset_generation_instruction"]["analyses"] = {"dataset_overview":{"dataset": args.dataset_name, "report": None}}

    return specs


def parse_commandline_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for building immuneML defintion YAML for Galaxy Create Dataset tool")
    parser.add_argument("-x", "--existing_dataset", choices=["True", "False"], default="False", help="Whether to use an already existing dataset from the current working directory (use the 'dataset_name' parameter to pass the name).")
    parser.add_argument("-d", "--dataset_name", default="dataset", help="The name of the dataset to import/export.")

    parser.add_argument("-r", "--format", help="The format of the repertoire/receptor dataset")
    parser.add_argument("-m", "--metadata_file", default="",
                        help="The metadata file when using a repertoire dataset. When using a receptor dataset, you may supply an empty string.")
    parser.add_argument("-i", "--is_repertoire", choices=["True", "False"], help="Whether to import a RepertoireDataset")
    parser.add_argument("-p", "--paired", choices=["True", "False"], default="False",
                        help="When the data is not repertoire data (metadata file = ''), this specifies whether the data is paired (ReceptorDataset) or unpaired (SequenceDataset)")
    parser.add_argument("-c", "--receptor_chains", choices=["TRA_TRB", "TRG_TRD", "IGH_IGL", "IGH_IGK"], default="TRA_TRB",
                        help="When the data is a ReceptorDataset, this specifies the type of receptor chains that are used.")
    parser.add_argument("-a", "--metadata_columns", default="", help="The name of metadata columns of a Sequence- or ReceptorDataset.")

    parser.add_argument("-l", "--label_name", default="", help="The label name to be used for reports.")
    parser.add_argument("-s", "--sequence_length_report", choices=["True", "False"], default="False", help="Whether to run the SequenceLengthDistribution report.")
    parser.add_argument("-g", "--vj_gene_report", choices=["True", "False"], default="False", help="Whether to run the VJGeneDistribution report.")
    parser.add_argument("-q", "--amino_acid_report", choices=["True", "False"], default="False", help="Whether to run the AminoAcidFrequencyDistribution report.")

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
