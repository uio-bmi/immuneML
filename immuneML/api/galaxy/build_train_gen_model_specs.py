import argparse
import sys
from pathlib import Path

from immuneML.api.galaxy.Util import Util
from immuneML.data_model.bnp_util import write_yaml
from immuneML.util.PathBuilder import PathBuilder


def parse_command_line_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for building specification for training generative "
                                                 "models in Galaxy")

    parser.add_argument("-e", "--gen_example_count", required=True,
                        help="Number of examples to generate.")
    parser.add_argument("-c", "--chain_type", required=True, choices=["TRA", "TRB", "IGH", "IGK", "IGL", "TRD", "TRG"],
                        help="Chain type of the sequences to generate.")
    parser.add_argument("-m", "--generative_method", choices=["SoNNia", "SimpleLSTM", "PWM", "SimpleVAE"],
                        required=True, help="Which generative model should be trained.")

    parser.add_argument("-s", "--sequence_length_report", choices=["True", "False"], default="False",
                        help="Whether to run the SequenceLengthDistribution report.")
    parser.add_argument("-q", "--amino_acid_report", choices=["True", "False"], default="False",
                        help="Whether to run the AminoAcidFrequencyDistribution report.")
    parser.add_argument("-k", "--kl_gen_model_report", choices=["True", "False"], default="False",
                        help="Whether to run the KLKmerComparison report.")

    parser.add_argument("-t", "--training_percentage", type=float, required=True,
                        help="The percentage of data used for training.")
    parser.add_argument("-x", "--export_dataset_type", choices=["generated_dataset", "combined_dataset"],
                        default="generated_dataset",
                        help="Whether to export only the generated dataset, or the combined training+(test+)generated dataset.")

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
    if parsed_args.kl_gen_model_report == "True":
        reports['kl_kmer_comparison'] = "KLKmerComparison"

    gen_model_args = {"locus": parsed_args.chain_type}

    if parsed_args.generative_method == "SimpleVAE":
        reports['generative_model_overview'] = "VAESummary"
        gen_model_args['region_type'] = 'imgt_cdr3'
    elif parsed_args.generative_method == "PWM":
        reports['generative_model_overview'] = "PWMSummary"
    elif parsed_args.generative_method == "SimpleLSTM":
        gen_model_args['region_type'] = 'imgt_cdr3'

    if parsed_args.generative_method == "SoNNia":
        gen_model_args["default_model_name"] = f"human{parsed_args.chain_type}"

    specs = {
        "definitions": {
            "datasets": {
                "dataset": {
                    "format": "AIRR",
                    "params": {"dataset_file": Util.discover_dataset_path()}
                }
            },
            "reports": reports,
            "ml_methods": {
                "generative_model": {
                    parsed_args.generative_method: gen_model_args
                }
            }
        },
        "instructions": {
            f"train_{parsed_args.generative_method}": {
                "type": "TrainGenModel",
                "dataset": "dataset",
                "method": "generative_model",
                "number_of_processes": 8,
                "gen_examples_count": int(parsed_args.gen_example_count),
                "training_percentage": float(parsed_args.training_percentage) / 100,
                "export_generated_dataset": True if parsed_args.export_dataset_type == "generated_dataset" else False,
                "export_combined_dataset": True if parsed_args.export_dataset_type == "combined_dataset" else False,
                "reports": list(reports.keys()),
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
