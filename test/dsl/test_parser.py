import shutil
from unittest import TestCase

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.data_model.dataset.Dataset import Dataset
from source.dsl.Parser import Parser
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestParser(TestCase):
    def test_parse_yaml_file(self):
        path = EnvironmentSettings.root_path + "test/tmp/parser/"
        dataset = Dataset(filenames=RepertoireBuilder.build([["AAA", "CCC"], ["TTTT"]], path, {"default": [1, 2]}),
                          params={"default": [1, 2]})
        PickleExporter.export(dataset, path, "dataset.pkl")

        spec = {
            "dataset_import": {
                "d1": {
                    "format": "Pickle",
                    "path": path + "dataset.pkl",
                    "result_path": path
                }
            },
            "encodings": {
                "a1": {
                    "dataset": "d1",
                    "type": "Word2Vec",
                    "params": {
                        "k": 3,
                        "model_creator": "sequence",
                        "size": 8,
                    }
                }
            },
            "ml_methods": {
                "simpleLR": {
                    "assessment_type": "LOOCV",
                    "type": "SimpleLogisticRegression",
                    "params": {
                        "penalty": "l1"
                    },
                    "encoding": "a1",
                    "labels": ["CD"],
                    "metrics": ["accuracy", "balanced_accuracy"],
                    "split_count": 1,
                    "model_selection_cv": False,
                    "model_selection_n_folds": -1,
                    "min_example_count": 3
                }
            },
            "reports": {
                "rep1": {
                    "type": "SequenceLengthDistribution",
                    "params": {
                        "dataset": "d1",
                        "batch_size": 3
                    }
                }
            },
            "simulation": {
                "motifs": {
                    "motif1": {
                        "seed": "CAS",
                        "instantiation": "Identity"
                    }
                },
                "signals": {
                    "signal1": {
                        "motifs": ["motif1"],
                        "implanting": "healthy_sequences"
                    }
                },
                "implanting": {
                    "var1": {
                        "signals": ["signal1"],
                        "repertoires": 0.4,
                        "sequences": 0.01
                    }
                }
            }
        }

        PathBuilder.build(path)

        with open(path + "tmp_yaml_spec.yaml", "w") as file:
            yaml.dump(spec, file, default_flow_style=False)

        symbol_table, _ = Parser.parse_yaml_file(path + "tmp_yaml_spec.yaml")

        self.assertTrue(all([symbol_table.contains(key) for key in
                             ["motif1", "signal1", "simpleLR", "rep1", "a1", "d1", "var1"]]))
        self.assertTrue(isinstance(symbol_table.get("d1")["dataset"], Dataset))

        shutil.rmtree(path)
