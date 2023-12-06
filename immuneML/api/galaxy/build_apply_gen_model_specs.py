import shutil

import argparse
import sys
from pathlib import Path

from immuneML.data_model.bnp_util import write_yaml, read_yaml
from immuneML.util.PathBuilder import PathBuilder


def parse_command_line_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for building specification for applying trained generative "
                                                 "model in Galaxy")

    parser.add_argument("-m", "--ml_config_path", required=True,
                        help="Path to the exported ml_config zip file with the trained model.")
    parser.add_argument("-e", "--gen_example_count", required=True,
                        help="Number of examples to generate.")
    parser.add_argument("-s", "--sequence_length_report", choices=["True", "False"], default="False",
                        help="Whether to run the SequenceLengthDistribution report.")
    parser.add_argument("-q", "--amino_acid_report", choices=["True", "False"], default="False",
                        help="Whether to run the AminoAcidFrequencyDistribution report.")
    parser.add_argument("-g", "--gen_model_overview", choices=["True", "False"], default="False",
                        help="Whether to run the model report.")
    parser.add_argument("-o", "--output_path", required=True,
                        help="Output location for the generated yaml file (directory).")
    parser.add_argument("-f", "--file_name", default="specs.yaml",
                        help="Output file name for the yaml file. Default name is 'specs.yaml' if not specified.")

    return parser.parse_args(args)


def build_specs(parsed_args):
    reports = {}
    if parsed_args.sequence_length_report == "True":
        reports['sequence_length_distribution'] = "SequenceLengthDistribution"
    if parsed_args.amino_acid_report == "True":
        reports['amino_acid_frequency_distribution'] = "AminoAcidFrequencyDistribution"

    model_overview_reports = {
        "PWM": "PWMSummary",
        "SimpleVAE": "VAESummary"
    }

    if parsed_args.gen_model_overview == "True":
        shutil.unpack_archive(parsed_args.ml_config_path, "./unpacked", 'zip')
        model_overview = read_yaml(Path('unpacked/model_overview.yaml'))
        if model_overview['type'] in model_overview_reports:
            reports['generative_model_overview'] = model_overview_reports[model_overview['type']]
        shutil.rmtree("./unpacked")

    specs = {
        "definitions": {
            "reports": reports
        },
        "instructions": {
            "inst1": {
                "type": "ApplyGenModel",
                "gen_examples_count": int(parsed_args.gen_example_count),
                "reports": list(reports.keys()),
                "ml_config_path": parsed_args.ml_config_path,
            }
        }
    }

    return specs


def main(args):
    parsed_args = parse_command_line_arguments(args)
    specs = build_specs(parsed_args)

    PathBuilder.build(parsed_args.output_path)
    output_location = Path(parsed_args.output_path) / parsed_args.file_name

    write_yaml(output_location, specs)

    return str(output_location)


if __name__ == "__main__":
    main(sys.argv[1:])
