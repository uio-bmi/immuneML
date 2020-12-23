import os
import shutil
from unittest import TestCase

import pandas as pd
import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app.ImmuneMLApp import ImmuneMLApp
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestSequenceAbundanceEncoding(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encoding(self):

        path = EnvironmentSettings.tmp_test_path + "integration_test_emerson_encoding/"
        PathBuilder.build(path)

        ref_path = path + "reference.csv"
        pd.DataFrame({"sequence_aas": ["GGG", "III", "TTT", "EFEF"], "v_alleles": ["TRBV6-1*01", "TRBV6-1*01", "TRBV6-1*01", "TRBV6-1*01"], 'j_alleles': ["TRBJ2-7", "TRBJ2-7", "TRBJ2-7", "TRBJ2-7"]}).to_csv(ref_path, index=False)

        repertoires, metadata = RepertoireBuilder.build([["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"],
                                                         ["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"],["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"],["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"]
                                                         ],
                                                        labels={"l1": [True, True, False, False, True, True, False, False, True, True, False, False,
                                                                       True, True, False, False]}, path=path)

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, params={"l1": [True, False]})
        PickleExporter.export(dataset, path)

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "Pickle",
                        "params": {
                            "path": path + f"{dataset.name}.iml_dataset",
                        }
                    }
                },
                "encodings": {
                    "e1": {
                        "SequenceAbundance": {
                            'comparison_attributes': ["sequence_aas", "v_alleles", "j_alleles"]
                        }
                    }
                },
                "ml_methods": {
                    "knn": {
                        "KNN": {
                            "n_neighbors": 1
                        },
                    }
                },
                "reports": {
                    "r1": {"ReferenceSequenceOverlap": {"reference_path": ref_path,
                                                        'comparison_attributes': ["sequence_aas", "v_alleles", "j_alleles"]}}
                }
            },
            "instructions": {
                "inst1": {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "encoding": "e1",
                            "ml_method": "knn"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "reports": {
                        }
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                    },
                    "labels": [{"l1": {"positive_class": True}}],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "number_of_processes": 2,
                    "reports": ["r1"],
                    "optimization_metric": "balanced_accuracy",
                    "refit_optimal_model": True,
                    "store_encoded_data": False
                }
            }
        }

        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_file, path + "result/")
        app.run()

        shutil.rmtree(path)
