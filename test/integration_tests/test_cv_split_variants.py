import os
import shutil
from unittest import TestCase

import pandas as pd
import yaml

from source.app.ImmuneMLApp import ImmuneMLApp
from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestCVSplitVariants(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def build_specs(self, path) -> dict:
        train_metadata_df = pd.DataFrame({"subject_id": [f"rep_{i}" for i in range(35)]})
        train_metadata_df.to_csv(path + "train.csv")
        train_metadata_df = pd.DataFrame({"subject_id": [f"rep_{i}" for i in range(36, 50)]})
        train_metadata_df.to_csv(path + "test.csv")
        return {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "repertoire_count": 50,
                            "result_path": path,
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
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                    },
                    "assessment": {
                        "split_strategy": "manual",
                        "manual_config": {
                            "train_metadata_path": path + "train.csv",
                            'test_metadata_path': path + "test.csv"
                        }
                    },
                    "labels": ["cmv"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "number_of_processes": 4,
                    "reports": None,
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": False,
                    "store_encoded_data": False
                }
            }
        }

    def test_dataset_generation(self):

        path = PathBuilder.build(EnvironmentSettings.tmp_test_path + "cv_split_variant/")
        repertoire_specs = self.build_specs(path)

        specs_filename = f"{path}specs.yaml"
        with open(specs_filename, "w") as file:
            yaml.dump(repertoire_specs, file)

        app = ImmuneMLApp(specs_filename, path + "result/")
        app.run()

        shutil.rmtree(path)
