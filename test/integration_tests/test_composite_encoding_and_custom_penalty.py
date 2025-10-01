import os
import shutil
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
                            'repertoire_count': 100,
                            'sequence_length_probabilities': {
                                5: 1.
                            },
                            'sequence_count_probabilities': {
                                10: 1.
                            },
                            "labels": {
                                "cmv": {
                                    True: 0.5,
                                    False: 0.5,
                                },
                                "batch": {
                                    'b1': 0.4,
                                    'b2': 0.6
                                },
                                'hla': {
                                    'a1': 0.25,
                                    'a2': 0.25,
                                    'a3': 0.25,
                                    'a4': 0.25,
                                }
                            }
                        }
                    }
                },
                "encodings": {
                    'composite': {
                        'Composite': {
                            'encoders': [
                                {
                                    "KmerFrequency": {
                                        "k": 3
                                    }
                                },
                                {
                                    'Metadata': {
                                        'metadata_fields': ['hla', 'batch']
                                    }
                                }
                            ]
                        }
                    },
                },
                "ml_methods": {
                    "logistic_regression": {
                        "LogRegressionCustomPenalty": {
                            "alpha": 1,
                            'n_lambda': 100,
                            'non_penalized_encodings': ['MetadataEncoder'],
                        }
                    }
                },

            },
            "instructions": {
                "train_test_instruction": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "encoding": "composite",
                            "ml_method": "logistic_regression"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7
                    },
                    "labels": ['cmv'],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ['precision', 'recall'],
                    "number_of_processes": 4,
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": False,
                }
            }
        }

    def test_dataset_generation(self):

        path = EnvironmentSettings.tmp_test_path / "composite_enc_custom_penalty/"

        specs = self.build_repertoire_specs(path)

        PathBuilder.remove_old_and_build(path)

        specs_filename = path / "specs.yaml"
        with specs_filename.open("w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_filename, path / "result/")
        app.run()
        shutil.rmtree(path)
