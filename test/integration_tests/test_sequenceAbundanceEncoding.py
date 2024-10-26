import os
import shutil
from unittest import TestCase

import pandas as pd
import yaml

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestSequenceAbundanceEncoding(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encoding(self):
        path = PathBuilder.remove_old_and_build(
            EnvironmentSettings.tmp_test_path / "integration_test_emerson_encoding/")

        ref_path = path / "reference.csv"
        pd.DataFrame({"cdr3_aa": ["GGG", "III", "TTT", "EFEF"],
                      "v_call": ["TRBV6-1*01", "TRBV6-1*01", "TRBV6-1*01", "TRBV6-1*01"],
                      'j_call': ["TRBJ2-7", "TRBJ2-7", "TRBJ2-7", "TRBJ2-7"]}).to_csv(ref_path, index=False)

        dataset = RepertoireBuilder.build_dataset(
            [["GGG", "III", "LLL", "MMM"], ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
             ["CCC", "FFF", "MMM"], ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"],
             ["GGG", "III", "LLL", "MMM"], ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
             ["CCC", "FFF", "MMM"], ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"],
             ["GGG", "III", "LLL", "MMM"], ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
             ["CCC", "FFF", "MMM"], ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"],
             ["GGG", "III", "LLL", "MMM"], ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
             ["CCC", "FFF", "MMM"], ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"]],
            labels={"l1": [True, True, False, False, True, True, False, False, True, True, False, False,
                           True, True, False, False]}, path=path)

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "AIRR",
                        "params": {
                            "dataset_file": str(dataset.dataset_file),
                        }
                    }
                },
                "encodings": {
                    "e1": {
                        "SequenceAbundance": {
                            'comparison_attributes': ["cdr3_aa", "v_call", "j_call"]
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
                    "r1": {"ReferenceSequenceOverlap": {"reference_path": str(ref_path),
                                                        'comparison_attributes': ["cdr3_aa", "v_call", "j_call"]}}
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
                }
            }
        }

        specs_file = path / "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_file, path / "result")
        app.run()

        shutil.rmtree(path)
