import os
import shutil
import sys

import yaml

from source.app.ImmuneMLApp import ImmuneMLApp
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class Quickstart:

    def create_specfication(self, path):

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "result_path": f'{path}dataset/',
                            "labels": {"CD": {True: 0.5, False: 0.5}}
                        }
                    }
                },
                "encodings": {
                    "e1": {
                        "Word2Vec": {
                            "k": 3,
                            "model_type": "sequence",
                            "vector_size": 8,
                        }
                    }
                },
                "ml_methods": {
                    "simpleLR": {
                        "SimpleLogisticRegression": {
                            "penalty": "l1"
                        },
                        "model_selection_cv": False,
                        "model_selection_n_folds": -1,
                    }
                },
                "preprocessing_sequences": {
                    "seq1": [
                        {"remove_duplicates": "DuplicateSequenceFilter"}
                    ]
                },
                "reports": {
                    "rep1": {
                        "SequenceLengthDistribution": {
                            "batch_size": 3
                        }
                    }
                }
            },
            "instructions": {
                "inst1": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "preprocessing": "seq1",
                            "encoding": "e1",
                            "ml_method": "simpleLR"
                        },
                        {
                            "preprocessing": "seq1",
                            "encoding": "e1",
                            "ml_method": "simpleLR"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "reports": {
                            "data_splits": []
                        }
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "reports": {
                            "data_splits": ["rep1"],
                            "models": [],
                        }
                    },
                    "labels": ["CD"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "reports": ["rep1"],
                    "batch_size": 10,
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": False
                }
            }
        }

        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        return specs_file

    def build_path(self, path: str = None):
        if path is None:
            path = EnvironmentSettings.root_path + "quickstart/"
            if os.path.isdir(path):
                shutil.rmtree(path)
            PathBuilder.build(path)
        return path

    def run(self, result_path: str):

        result_path = self.build_path(result_path)
        specs_file = self.create_specfication(result_path)

        app = ImmuneMLApp(specs_file, result_path + "quickstart/")
        app.run()


def main():
    quickstart = Quickstart()
    quickstart.run(sys.argv[1] if len(sys.argv) == 2 else None)


if __name__ == "__main__":
    main()
