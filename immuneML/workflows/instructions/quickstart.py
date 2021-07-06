import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class Quickstart:

    def create_specfication(self, path: Path):

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "AIRR",
                        "params": {
                            "path": str(path / "../synthetic_dataset/result/simulation_instruction/exported_dataset/airr/"),
                            "metadata_file": str(path / "../synthetic_dataset/result/simulation_instruction/exported_dataset/airr/metadata.csv")
                        }
                    }
                },
                "encodings": {
                    "e1": {
                        "KmerFrequency": {
                            "k": 3
                        }
                    },
                    "e2": {
                        "KmerFrequency": {
                            "k": 2
                        }
                    }
                },
                "ml_methods": {
                    "simpleLR": {
                        "LogisticRegression": {
                            "C": 0.1,
                            "penalty": "l1",
                            "max_iter": 200
                        }}
                },
                "reports": {
                    "rep1": {
                        "SequenceLengthDistribution": {
                            "batch_size": 3
                        }
                    },
                    "hprep": "MLSettingsPerformance",
                    "coef": "Coefficients"
                }
            },
            "instructions": {
                "machine_learning_instruction": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "encoding": "e1",
                            "ml_method": "simpleLR"
                        },
                        {
                            "encoding": "e2",
                            "ml_method": "simpleLR"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "reports": {
                            "data_splits": ["rep1"],
                            'models': ["coef"]
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
                    "labels": ["my_signal"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "reports": ["hprep"],
                    "number_of_processes": 3,
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": False
                }
            }
        }
        PathBuilder.build(path)
        specs_file = path / "specs.yaml"
        with specs_file.open("w") as file:
            yaml.dump(specs, file)

        return specs_file

    def build_path(self, path: str = None):
        if path is None:
            path = EnvironmentSettings.root_path / "quickstart/"
            if os.path.isdir(path):
                shutil.rmtree(path)
            PathBuilder.build(path)
        else:
            path = PathBuilder.build(path)
        return path

    def _simulate_dataset_with_signals(self, path: Path):

        print("immuneML quickstart: generating a synthetic dataset...")

        PathBuilder.build(path)

        specs = {
            "definitions": {
                "datasets": {
                    "my_synthetic_dataset": {"format": "RandomRepertoireDataset", "params": {"labels": {}}}
                },
                "motifs": {"my_motif": {"seed": "AA", "instantiation": "GappedKmer"}},
                "signals": {"my_signal": {"motifs": ["my_motif"], "implanting": "HealthySequence"}},
                "simulations": {"my_simulation": {"my_implantng": {"signals": ["my_signal"], "dataset_implanting_rate": 0.5,
                                                                   "repertoire_implanting_rate": 0.1}}}
            },
            "instructions": {"simulation_instruction": {"type": "Simulation", "dataset": "my_synthetic_dataset", "simulation": "my_simulation",
                                                        "export_formats": ["AIRR"]}}
        }

        specs_file = path / "simulation_specs.yaml"
        with specs_file.open("w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_file, path / "result")
        app.run()

        print("immuneML quickstart: finished generating a synthetic dataset.")

    def run(self, result_path: str):

        result_path = self.build_path(result_path)

        logging.basicConfig(filename=Path(result_path) / "log.txt", level=logging.ERROR, format='%(asctime)s %(levelname)s: %(message)s')
        warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logging.warning(message)

        self._simulate_dataset_with_signals(result_path / "synthetic_dataset")

        print("immuneML quickstart: training a machine learning model...")
        specs_file = self.create_specfication(result_path / "machine_learning_analysis")
        app = ImmuneMLApp(specs_file, result_path / "machine_learning_analysis/result")
        app.run()

        print("immuneML quickstart: finished training a machine learning model.")


def main():
    quickstart = Quickstart()
    quickstart.run(sys.argv[1] if len(sys.argv) == 2 else None)


if __name__ == "__main__":
    main()
