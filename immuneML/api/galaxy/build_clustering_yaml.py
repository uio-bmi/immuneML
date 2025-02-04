import argparse
import sys
from pathlib import Path

from immuneML.api.galaxy.Util import Util
from immuneML.data_model.bnp_util import write_yaml
from immuneML.util.PathBuilder import PathBuilder

from immuneML.ml_metrics.ClusteringMetric import INTERNAL_EVAL_METRICS, EXTERNAL_EVAL_METRICS


def build_labels(labels_str):
    labels = labels_str.split(",")
    return [label.strip().strip("'\"") for label in labels]


def parse_command_line_arguments(args):
    parser = argparse.ArgumentParser(
        description="Tool for building specification for applying previously trained ML models in Galaxy")

    parser.add_argument("-l", "--labels", type=str, default="",
                        help="Which metadata labels should be predicted for the dataset (separated by comma).")
    parser.add_argument("-e", "--eval_metrics", type=str, choices=INTERNAL_EVAL_METRICS + EXTERNAL_EVAL_METRICS,
                        default=[], nargs="+",
                        help="External evaluation metrics to use for clustering, for these metrics, clusters are compared to a provided label.")

    parser.add_argument("-k", "--encoding_k", type=int, required=True, help="")
    parser.add_argument("-n", "--n_clusters", type=int, required=True, help="")
    parser.add_argument("-d", "--dim_red_method", type=str, choices=["PCA", "UMAP", "TSNE", "None"], default="None",
                        help="External evaluation metrics to use for clustering, for these metrics, clusters are compared to a provided label.")
    parser.add_argument("-t", "--training_percentage", type=int, default=100)

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
            'reports': {
                'dim_reduction': {'DimensionalityReduction': {"dim_red_method": {}}}
            },
            'encodings': {
                'kmer': {
                    'KmerFrequency': {
                        "k": parsed_args.encoding_k
                    }
                }
            },
            'ml_methods': {
                'kmeans': {
                    'KMeans': {
                        'n_clusters': parsed_args.n_clusters
                    }
                },
                "pca_2": {
                    "PCA": {
                        "n_components": 2
                    }
                }
            }
        },
        "instructions": {
            f"clustering": {
                "type": "Clustering",
                "dataset": "dataset",
                'metrics': parsed_args.eval_metrics,
                'clustering_settings': [
                    {'encoding': 'kmer', "dim_reduction": "pca_2", 'method': 'kmeans'},
                ],
                'split_config': {
                    'split_strategy': 'random',
                    'training_percentage': parsed_args.training_percentage / 100
                }
            }
        }
    }

    if parsed_args.labels != "":
        specs["instructions"]["clustering"]["labels"] = build_labels(parsed_args.labels)
        specs['definitions']['reports']['dim_reduction']["DimensionalityReduction"]['label'] = (
            specs)["instructions"]["clustering"]["labels"][0]

    if parsed_args.dim_red_method in ["PCA", "TSNE", "UMAP"]:
        specs["definitions"]["ml_methods"][parsed_args.dim_red_method.lower()] = {
            parsed_args.dim_red_method: {"n_components": 2}}
        if parsed_args.dim_red_method == 'TSNE':
            specs["definitions"]["ml_methods"][parsed_args.dim_red_method.lower()][parsed_args.dim_red_method]['init'] = "random"
        specs["definitions"]["reports"]["dim_reduction"]["DimensionalityReduction"]["dim_red_method"] = {
            parsed_args.dim_red_method.upper(): {"n_components": 2}
        }
        specs["instructions"]["clustering"]["reports"] = ["dim_reduction"]

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
