import os
import shutil
from unittest import TestCase

import yaml

from source.app.ImmuneMLApp import ImmuneMLApp
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestExploratoryAnalysisDesignMatrixExporter(TestCase):

    def test_run(self):

        path = EnvironmentSettings.tmp_test_path + "random_dataset_workflow/"
        PathBuilder.build(path)
        os.environ["cache_type"] = "test"

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "result_path": path,
                            "repertoire_count": 100,
                            "sequence_count_probabilities": {
                                10: 0.5,
                                12: 0.5
                            },
                            "sequence_length_probabilities": {
                                5: 1.
                            },
                            "labels": {
                                "cmv": {
                                    True: 0.5,
                                    False: 0.5
                                }
                            }
                        }
                    }
                },
                "encodings": {
                    "kmer_freq": {
                        "KmerFrequency": {
                            "k": 3,
                            "sequence_encoding": "continuous_kmer",
                            "normalization_type": "relative_frequency",
                            "reads": "unique"
                        }
                    }
                },
                "ml_methods": {
                    "logistic_regression": {
                        "SimpleLogisticRegression": {
                            "C": 100,
                            "penalty": "l1"
                        }
                    }
                },

            },
            "instructions": {
                "train_test_instruction": {
                    "type": "HPOptimization",
                    "settings": [
                        {
                            "encoding": "kmer_freq",
                            "ml_method": "logistic_regression"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                    },
                    "labels": ["cmv"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "batch_size": 4,
                    "reports": None,
                    "optimization_metric": "balanced_accuracy"
                }
            }
        }

        specs_filename = f"{path}specs.yaml"
        with open(specs_filename, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_filename, path)
        app.run()

        shutil.rmtree(path)
