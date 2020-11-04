import os
import shutil
from argparse import Namespace
from unittest import TestCase

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app.ImmuneMLApp import run_immuneML
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.util.PathBuilder import PathBuilder


class TestGalaxyYamlTool(TestCase):
    def test_run(self):

        path = PathBuilder.build(f"{EnvironmentSettings.tmp_test_path}api_galaxy_yaml_tool/")
        result_path = f"{path}result/"

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(10, {10: 1}, {12: 1}, {}, result_path)
        dataset.name = "d1"
        PickleExporter.export(dataset, result_path)

        specs = {
            "definitions": {
                "datasets": {
                    "new_d1": {
                        "format": "Pickle",
                        "params": {
                            "metadata_file": f"{result_path}d1_metadata.csv"
                        }
                    },
                    "d2": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "repertoire_count": 50,
                            "sequence_length_probabilities": {10: 1},
                            'sequence_count_probabilities': {10: 1},
                            'labels': {
                                "CD": {
                                    True: 0.5,
                                    False: 0.5
                                }
                            }
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
                    },
                    "e2": {
                        "Word2Vec": {
                            "k": 3,
                            "model_type": "sequence",
                            "vector_size": 10,
                        }
                    },
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
            },
            "instructions": {
                "inst1": {
                    "type": "DatasetGeneration",
                    "datasets": ["new_d1", 'd2'],
                    "export_formats": ["AIRR"]
                },
                "inst2": {
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
                        "training_percentage": 0.7
                    },
                    "selection": {
                        "split_strategy": "random",
                        "split_count": 2,
                        "training_percentage": 0.7
                    },
                    "labels": ["CD"],
                    "dataset": "d2",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy", "auc"],
                    "reports": [],
                    "batch_size": 10,
                    "optimization_metric": "accuracy",
                    'refit_optimal_model': False,
                    "store_encoded_data": False
                }
            }
        }

        specs_path = f"{path}specs.yaml"
        with open(specs_path, "w") as file:
            yaml.dump(specs, file)

        run_immuneML(Namespace(**{"specification_path": specs_path, "result_path": result_path + 'result/', 'tool': "GalaxyYamlTool"}))

        self.assertTrue(os.path.exists(f"{result_path}result/inst1/new_d1/AIRR"))
        self.assertTrue(os.path.exists(f"{result_path}result/inst1/d2/AIRR"))
        self.assertTrue(os.path.exists(f"{result_path}result/d2"))

        shutil.rmtree(path)
