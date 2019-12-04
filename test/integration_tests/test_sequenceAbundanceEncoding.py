import shutil
from unittest import TestCase

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app.ImmuneMLApp import ImmuneMLApp
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestSequenceAbundanceEncoding(TestCase):

    def test_encoding(self):

        path = EnvironmentSettings.tmp_test_path + "integration_test_emerson_encoding/"
        PathBuilder.build(path)

        filenames, metadata = RepertoireBuilder.build([["GGG", "III", "LLL", "MMM"],
                                                       ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                       ["CCC", "FFF", "MMM"],
                                                       ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"]],
                                                      labels={"l1": [True, True, False, False]}, path=path)

        dataset = RepertoireDataset(filenames=filenames, metadata_file=metadata, identifier="1", params={"l1": [True, False]})
        PickleExporter.export(dataset, path, "dataset.pickle")

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "Pickle",
                        "path": path + "dataset.pickle",
                        "params": {
                            "result_path": path
                        }
                    }
                },
                "encodings": {
                    "e1": {
                        "type": "SequenceAbundance",
                        "params": {}
                    }
                },
                "ml_methods": {
                    "knn": {
                        "type": "KNN",
                        "params": {
                            "n_neighbors": 1
                        },
                    }
                }
            },
            "instructions": {
                "inst1": {
                    "type": "HPOptimization",
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
                        "label_to_balance": None,
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "label_to_balance": None,
                    },
                    "labels": ["l1"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy"],
                    "batch_size": 2
                }
            }
        }

        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_file, path)
        app.run()

        shutil.rmtree(path)
