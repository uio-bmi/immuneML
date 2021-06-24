import os
import shutil
from pathlib import Path
from unittest import TestCase

import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestRandomDatasetWorkflow(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def build_repertoire_specs(self, path) -> dict:
        return {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "result_path": str(path),
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
                        "LogisticRegression": {
                            "C": 100,
                            "penalty": "l1"
                        }
                    }
                },

            },
            "instructions": {
                "train_test_instruction": {
                    "type": "TrainMLModel",
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
                        "split_strategy": "stratified_k_fold",
                        "split_count": 5
                    },
                    "labels": ["cmv"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "number_of_processes": 4,
                    "reports": None,
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": False,
                }
            }
        }

    def build_receptor_specs(self, path: str) -> dict:
        return {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomReceptorDataset",
                        "params": {
                            "result_path": str(path),
                            "receptor_count": 500,
                            "chain_1_length_probabilities": {
                                5: 1.
                            },
                            "chain_2_length_probabilities": {
                                6: 1.
                            },
                            "labels": {
                                "cmv_epitope": {
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
                        "LogisticRegression": {
                            "C": 100,
                            "penalty": "l1"
                        }
                    }
                },

            },
            "instructions": {
                "train_test_instruction": {
                    "type": "TrainMLModel",
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
                    "labels": ["cmv_epitope"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "number_of_processes": 4,
                    "reports": None,
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": True,
                }
            }
        }

    def test_dataset_generation(self):

        path = EnvironmentSettings.tmp_test_path / "random_dataset_workflow/"

        repertoire_specs = self.build_repertoire_specs(path)
        self.run_example(repertoire_specs, path)

        # receptor_specs = self.build_receptor_specs(path)
        # self.run_example(receptor_specs, path)

    def run_example(self, specs: dict, path: Path):

        PathBuilder.build(path)

        specs_filename = path / "specs.yaml"
        with specs_filename.open("w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_filename, path / "result/")
        app.run()

        shutil.rmtree(path)
