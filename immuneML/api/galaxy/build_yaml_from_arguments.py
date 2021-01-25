import argparse
import glob
import itertools as it
import logging
import sys
import warnings
from pathlib import Path

import yaml

from immuneML.encodings.kmer_frequency.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.reports.ml_reports.CoefficientPlottingSetting import CoefficientPlottingSetting
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler


def get_sequence_enc_type(sequence_type, position_type, gap_type):
    if sequence_type == "complete":
        encoding_type = SequenceEncodingType.IDENTITY
    else:
        if position_type == "positional":
            if gap_type == "gapped":
                encoding_type = SequenceEncodingType.IMGT_GAPPED_KMER
            else:
                encoding_type = SequenceEncodingType.IMGT_CONTINUOUS_KMER
        else:
            if gap_type == "gapped":
                encoding_type = SequenceEncodingType.GAPPED_KMER
            else:
                encoding_type = SequenceEncodingType.CONTINUOUS_KMER

    return encoding_type.name


def build_encodings_specs(args):
    encodings = dict()

    for i in range(len(args.sequence_type)):
        enc_name = f"encoding_{i + 1}"
        enc_spec = dict()

        enc_spec["sequence_encoding"] = get_sequence_enc_type(args.sequence_type[i],
                                                              None if args.position_type is None else args.position_type[i],
                                                              None if args.gap_type is None else args.gap_type[i])
        enc_spec["reads"] = args.reads[i]

        if args.sequence_type[i] == "subsequence":
            if args.gap_type[i] == "gapped":
                enc_spec["k_left"] = args.k_left[i]
                enc_spec["k_right"] = args.k_right[i]
                enc_spec["min_gap"] = args.min_gap[i]
                enc_spec["max_gap"] = args.max_gap[i]
            else:
                enc_spec["k"] = args.k[i]

        encodings[enc_name] = {"KmerFrequency": enc_spec}

    return encodings


def get_ml_method_spec(ml_method_class, model_selection_n_folds=5):
    if ml_method_class == "LogisticRegression" or ml_method_class == "SimpleLogisticRegression":
        ml_spec = {
            "logistic_regression": {
                "LogisticRegression": {
                    "penalty": ["l1"],
                    "C": [0.01, 0.1, 1, 10, 100],
                    "class_weight": ["balanced"],
                    "show_warnings": False
                },
                "model_selection_cv": True,
                "model_selection_n_folds": model_selection_n_folds
            }
        }
    elif ml_method_class == "RandomForestClassifier":
        ml_spec = {
            "random_forest": {
                "RandomForestClassifier": {
                    "n_estimators": [10, 50, 100],
                    "class_weight": ["balanced"],
                    "show_warnings": False
                },
                "model_selection_cv": True,
                "model_selection_n_folds": model_selection_n_folds
            }
        }
    elif ml_method_class == "SVM":
        ml_spec = {
            "support_vector_machine": {
                "SVM": {
                    "penalty": ["l1"],
                    "C": [0.01, 0.1, 1, 10, 100],
                    "class_weight": ["balanced"],
                    "show_warnings": False
                },
                "model_selection_cv": True,
                "model_selection_n_folds": model_selection_n_folds
            }
        }
    elif ml_method_class == "KNN":
        ml_spec = {
            "k_nearest_neighbors": {
                "KNN": {
                    "n_neighbors": [3, 5, 10],
                    "show_warnings": False
                },
                "model_selection_cv": True,
                "model_selection_n_folds": model_selection_n_folds
            }
        }
    else:
        ml_spec = {ml_method_class: ml_method_class}

    return ml_spec


def build_ml_methods_specs(args):
    ml_methods_spec = dict()

    for method in args.ml_methods:
        ml_methods_spec.update(get_ml_method_spec(method))

    return ml_methods_spec


def build_settings_specs(enc_names, ml_names):
    return [{"encoding": enc_name, "ml_method": ml_name} for enc_name, ml_name in it.product(enc_names, ml_names)]


def discover_dataset_params():
    dataset = glob.glob("*.iml_dataset")

    assert len(dataset) > 0, "no .iml_dataset file was present in the current working directory"
    assert len(dataset) < 2, "multiple .iml_dataset files were present in the current working directory"

    dataset_path = dataset[0]

    dataset_name = dataset_path.rsplit('.iml_dataset', 1)[0]

    return {"path": dataset_path,
            "metadata_file": f"{dataset_name}_metadata.csv"}


def build_labels(labels_str):
    labels = labels_str.split(",")
    return [label.strip().strip("'\"") for label in labels]


def build_specs(args):
    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    "format": "Pickle",
                    "params": None
                }
            },
            "encodings": dict(),
            "ml_methods": dict(),
            "reports": {
                "coefficients": {
                    "Coefficients": {
                        "coefs_to_plot": [CoefficientPlottingSetting.N_LARGEST.name],
                        "n_largest": [25]
                    }
                },
                "benchmark": "MLSettingsPerformance"
            }
        },
        "instructions": {
            "inst1": {
                "type": "TrainMLModel",
                "settings": [],
                "assessment": {
                    "split_strategy": "random",
                    "split_count": None,
                    "training_percentage": None,
                    "reports": {
                        "models": ["coefficients"]
                    }
                },
                "selection": {
                    "split_strategy": "random",
                    "split_count": 1,
                    "training_percentage": 0.7,
                },
                "labels": [],
                "dataset": "d1",
                "strategy": "GridSearch",
                "metrics": ["accuracy", "balanced_accuracy"],
                "number_of_processes": 10,
                "reports": ["benchmark"],
                "optimization_metric": "balanced_accuracy",
                'refit_optimal_model': True,
                "store_encoded_data": False
            }
        }
    }

    enc_specs = build_encodings_specs(args)
    ml_specs = build_ml_methods_specs(args)
    settings_specs = build_settings_specs(enc_specs.keys(), ml_specs.keys())
    dataset_params = discover_dataset_params()
    labels = build_labels(args.labels)

    specs["definitions"]["datasets"]["d1"]["params"] = dataset_params
    specs["definitions"]["encodings"] = enc_specs
    specs["definitions"]["ml_methods"] = ml_specs
    specs["instructions"]["inst1"]["settings"] = settings_specs
    specs["instructions"]["inst1"]["assessment"]["split_count"] = args.split_count
    specs["instructions"]["inst1"]["assessment"]["training_percentage"] = args.training_percentage / 100
    specs["instructions"]["inst1"]["labels"] = labels

    return specs


def check_arguments(args):
    assert 100 >= args.training_percentage >= 10, "training_percentage must range between 10 and 100"
    assert args.split_count >= 1, "The minimal split_count is 1."

    encoding_err = "When multiple encodings are used, fields must still be of equal length, add 'NA' variables where necessary"
    assert len(args.sequence_type) == len(args.reads), encoding_err
    assert args.position_type is None or len(args.sequence_type) == len(args.position_type), encoding_err
    assert args.gap_type is None or len(args.sequence_type) == len(args.gap_type), encoding_err
    assert args.k is None or len(args.sequence_type) == len(args.k), encoding_err
    assert args.k_left is None or len(args.sequence_type) == len(args.k_left), encoding_err
    assert args.k_right is None or len(args.sequence_type) == len(args.k_right), encoding_err
    assert args.min_gap is None or len(args.sequence_type) == len(args.min_gap), encoding_err
    assert args.max_gap is None or len(args.sequence_type) == len(args.max_gap), encoding_err


def parse_commandline_arguments(args):
    ReflectionHandler.get_classes_by_partial_name("", "ml_methods/")
    ml_method_names = [cl.__name__ for cl in ReflectionHandler.all_nonabstract_subclasses(MLMethod)] + ["SimpleLogisticRegression"]

    parser = argparse.ArgumentParser(description="tool for building immuneML Galaxy YAML from arguments")
    parser.add_argument("-o", "--output_path", required=True, help="Output location for the generated yaml file (directiory).")
    parser.add_argument("-f", "--file_name", default="specs.yaml",
                        help="Output file name for the yaml file. Default name is 'specs.yaml' if not specified.")
    parser.add_argument("-l", "--labels", required=True,
                        help="Which metadata labels should be predicted for the dataset (separated by comma).")
    parser.add_argument("-m", "--ml_methods", nargs="+", choices=ml_method_names, required=True,
                        help="Which machine learning methods should be applied.")
    parser.add_argument("-t", "--training_percentage", type=float, required=True,
                        help="The percentage of data used for training.")
    parser.add_argument("-c", "--split_count", type=int, required=True,
                        help="The number of times to repeat the training process with a different random split of the data.")
    parser.add_argument("-s", "--sequence_type", choices=["complete", "subsequence"], default=["subsequence"], nargs="+",
                        help="Whether complete CDR3 sequences are used, or k-mer subsequences.")
    parser.add_argument("-p", "--position_type", choices=["invariant", "positional"], nargs="+",
                        help="Whether IMGT-positional information is used for k-mers, or the k-mer positions are position-invariant.")
    parser.add_argument("-g", "--gap_type", choices=["gapped", "ungapped"], nargs="+", help="Whether the k-mers contain gaps.")
    parser.add_argument("-k", "--k", type=int, nargs="+", help="K-mer size.")
    parser.add_argument("-kl", "--k_left", type=int, nargs="+", help="Length before gap when k-mers are used.")
    parser.add_argument("-kr", "--k_right", type=int, nargs="+", help="Length after gap when k-mers are used.")
    parser.add_argument("-gi", "--min_gap", type=int, nargs="+", help="Minimal gap length when gapped k-mers are used.")
    parser.add_argument("-ga", "--max_gap", type=int, nargs="+", help="Maximal gap length when gapped k-mers are used.")
    parser.add_argument("-r", "--reads", choices=[ReadsType.UNIQUE.value, ReadsType.ALL.value], nargs="+", default=[ReadsType.UNIQUE.value],
                        help="Whether k-mer counts should be scaled by unique clonotypes or all observed receptor sequences")

    return parser.parse_args(args)


def main(args):
    logging.basicConfig(filename="build_yaml_from_args_log.txt", level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logging.warning(message)

    parsed_args = parse_commandline_arguments(args)
    check_arguments(parsed_args)

    specs = build_specs(parsed_args)

    PathBuilder.build(parsed_args.output_path)
    output_location = Path(parsed_args.output_path) / parsed_args.file_name

    with output_location.open("w") as file:
        yaml.dump(specs, file)

    return output_location


if __name__ == "__main__":
    main(sys.argv[1:])
