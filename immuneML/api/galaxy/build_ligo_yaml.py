import argparse
import sys
from pathlib import Path

from immuneML.data_model.bnp_util import write_yaml
from immuneML.util.PathBuilder import PathBuilder


def parse_command_line_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for building specification for simulating data with LiGO in Galaxy.")

    parser.add_argument("-m", "--motif_seed", required=True,
                        help="Seed of the motif to implant.")
    parser.add_argument("-e", "--example_with_motif_count", required=True, type=int,
                        help="Number of examples to generate containing the motif.")
    parser.add_argument("-w", "--example_without_motif_count", required=True, type=int,
                        help="Number of examples to generate NOT containing the motif.")
    parser.add_argument("-s", "--simulation_strategy",  choices=["RejectionSampling", "Implanting"], required=True,
                        help="Strategy for simulating signals into sequences.")
    parser.add_argument("-d", "--dataset_type",  choices=["repertoire", "sequence"], required=True,
                        help="Type of dataset to generate (RepertoireDataset or SequenceDataset, ReceptorDataset is currently not available)")
    parser.add_argument("-c", "--chain_type",  choices=["humanTRA", "humanTRB", "humanIGH", "humanIGK", "humanIGL"], required=True,
                        help="Chain type for the simulated dataset, to be used for default OLGA model selection.")
    parser.add_argument("-p", "--signal_percentage",  type=float, default=100,
                        help="Percentage of sequences that contain the signal.")
    parser.add_argument("-r", "--repertoire_size",  type=int, required=False,
                        help="Number of sequences per repertoire, if dataset_type is repertoire.")

    parser.add_argument("-o", "--output_path", required=True,
                        help="Output location for the generated yaml file (directory).")
    parser.add_argument("-f", "--file_name", default="specs.yaml",
                        help="Output file name for the yaml file. Default name is 'specs.yaml' if not specified.")

    return parser.parse_args(args)


def build_specs(parsed_args):

    specs = {
        "definitions": {
            "motifs": {
                "motif1": {
                    "seed": parsed_args.motif_seed
                }
            },
            "signals": {
                "signal1": {
                    "motifs": ["motif1"]
                }
            },
            "simulations": {
                "sim1": {
                    "sim_items": {
                        "signal": {
                            "generative_model": {
                                "default_model_name": parsed_args.chain_type,
                                "type": "OLGA"
                            },
                            "number_of_examples": parsed_args.example_with_motif_count,
                            "signals": {"signal1": parsed_args.signal_percentage / 100},
                            "seed": 100,
                            "receptors_in_repertoire_count": parsed_args.repertoire_size if parsed_args.dataset_type == "repertoire" else None
                        },
                        "no_signal": {
                            "generative_model": {
                                "default_model_name": parsed_args.chain_type,
                                "type": "OLGA"
                            },
                            "number_of_examples": parsed_args.example_without_motif_count,
                            "signals": {},
                            "seed": 200,
                            "receptors_in_repertoire_count": parsed_args.repertoire_size if parsed_args.dataset_type == "repertoire" else None
                        }
                    },
                    "is_repertoire": True if parsed_args.dataset_type == "repertoire" else False,
                    "sequence_type": "amino_acid",
                    "simulation_strategy": parsed_args.simulation_strategy
                }
            }
        },
        "instructions": {
            f"simulate_with_ligo": {
                "type": "LigoSim",
                "simulation": "sim1",
                "number_of_processes": 8,
                "export_p_gens": False,
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
