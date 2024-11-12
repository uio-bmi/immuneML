import argparse
import sys
from pathlib import Path

import yaml
import os

from immuneML.api.galaxy.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from immuneML.data_model.SequenceParams import RegionType



def get_dataset_specs(args):
    dataset_specs = {"format": args.format,
                     "params": {"region_type": RegionType.IMGT_CDR3.name,
                                "result_path": "./",
                                "path": "./"}}

    if args.is_repertoire == "True":
        dataset_specs["params"]["is_repertoire"] = True
        dataset_specs["params"]["metadata_file"] = args.metadata_file
    else:
        dataset_specs["params"]["is_repertoire"] = False
        if args.metadata_columns != "":
            dataset_specs["params"]["label_columns"] = args.metadata_columns.split(",")

        paired = True if args.paired == "True" else False

        dataset_specs["params"]["paired"] = paired
        if paired:
            dataset_specs["params"]["receptor_chains"] = args.receptor_chains

    return {"dataset": dataset_specs}



def add_report_with_label(specs, args, report_name, report_key):
    if args.label_name != "":
        specs["definitions"]["reports"][f"{report_key}_report"] = {report_name: {"label": args.label_name}}
    else:
        specs["definitions"]["reports"][f"{report_key}_report"] = report_name

    specs["instructions"]["my_dataset_generation_instruction"]["analyses"][f"{report_key}_analysis"] = {
        "dataset": "dataset", "report": f"{report_key}_report"}


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
        specs["definitions"]["datasets"]["dataset"] = {"format": "AIRR", "params": {"dataset_file": Util.discover_dataset_path()}}
    else:
        specs["definitions"]["datasets"] = get_dataset_specs(args)

    if args.sequence_length_report == "True":
        specs["definitions"]["reports"]["sequence_length_report"] = "SequenceLengthDistribution"
        specs["instructions"]["my_dataset_generation_instruction"]["analyses"]["sequence_length_analysis"] = {
            "dataset": "dataset", "report": "sequence_length_report"}

    if args.sequence_count_report == "True":
        add_report_with_label(specs, args, report_name="SequenceCountDistribution", report_key="sequence_count")

        # Note: if 'existing' dataset is used, it is not known beforehand whether this is a repertoire dataset
        # however, 'is_repertoire' is True by default in Galaxy, and even when running the repertoire report
        # on a non-repertoire dataset, the worst thing that happens is that the report fails with a warning
        if args.is_repertoire == "True":
            add_report_with_label(specs, args, report_name="RepertoireClonotypeSummary", report_key="repertoire_clone_count")


    if args.vj_gene_report == "True":
        add_report_with_label(specs, args, report_name="VJGeneDistribution", report_key="vj_gene")

    if args.amino_acid_report == "True":
        add_report_with_label(specs, args, report_name="AminoAcidFrequencyDistribution", report_key="amino_acid")

    if len(specs["instructions"]["my_dataset_generation_instruction"]["analyses"]) == 0:
        specs["instructions"]["my_dataset_generation_instruction"]["analyses"] = {
            "dataset_overview": {"dataset": "dataset", "report": None}}

    return specs


def parse_commandline_arguments(args):
    parser = argparse.ArgumentParser(
        description="Tool for building immuneML defintion YAML for Galaxy Create Dataset tool")
    parser.add_argument("-x", "--existing_dataset", choices=["True", "False"], default="False",
                        help="Whether to use an already existing dataset from the current working directory (use the 'dataset_name' parameter to pass the name).")

    parser.add_argument("-r", "--format", help="The format of the repertoire/receptor dataset")
    parser.add_argument("-m", "--metadata_file", default="",
                        help="The metadata file when using a repertoire dataset. When using a receptor dataset, you may supply an empty string.")
    parser.add_argument("-i", "--is_repertoire", choices=["True", "False"],
                        help="Whether to import a RepertoireDataset")
    parser.add_argument("-p", "--paired", choices=["True", "False"], default="False",
                        help="When the data is not repertoire data (metadata file = ''), this specifies whether the data is paired (ReceptorDataset) or unpaired (SequenceDataset)")
    parser.add_argument("-c", "--receptor_chains", choices=["TRA_TRB", "TRG_TRD", "IGH_IGL", "IGH_IGK"],
                        default="TRA_TRB",
                        help="When the data is a ReceptorDataset, this specifies the type of receptor chains that are used.")
    parser.add_argument("-a", "--metadata_columns", default="",
                        help="The name of metadata columns of a Sequence- or ReceptorDataset.")

    parser.add_argument("-l", "--label_name", default="", help="The label name to be used for reports.")
    parser.add_argument("-s", "--sequence_length_report", choices=["True", "False"], default="False",
                        help="Whether to run the SequenceLengthDistribution report.")
    parser.add_argument("-u", "--sequence_count_report", choices=["True", "False"], default="False",
                        help="Whether to run the SequenceCountDistribution report.")

    parser.add_argument("-g", "--vj_gene_report", choices=["True", "False"], default="False",
                        help="Whether to run the VJGeneDistribution report.")
    parser.add_argument("-q", "--amino_acid_report", choices=["True", "False"], default="False",
                        help="Whether to run the AminoAcidFrequencyDistribution report.")

    parser.add_argument("-o", "--output_path", required=True,
                        help="Output location for the generated yaml file (directiory).")
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
