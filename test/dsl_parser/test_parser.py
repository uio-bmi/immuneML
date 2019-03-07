import os
from unittest import TestCase

import yaml

from source.dsl_parsers.Parser import Parser
from source.ml_methods.MLMethod import MLMethod
from source.simulation.implants.Signal import Signal


class TestParser(TestCase):
    def test_parse(self):
        parsed = Parser.parse({
            "repertoire_count": 200,
            "sequence_count": 50,
            "ml_methods": ["LogisticRegression", "SVM", "RandomForestClassifier"],
            "encoder": "KmerFrequencyEncoder",
            "encoder_params": {
                "normalization_type": "relative_frequency",
                "reads": "unique",
                "sequence_encoding": "identity",
                "k": 3
            },
            "simulation": {
                "motifs": [
                    {
                        "id": "motif1",
                        "seed": "CAS",
                        "instantiation": "identity"
                    }
                ],
                "signals": [
                    {
                        "id": "signal1",
                        "motifs": ["motif1"],
                        "implanting": "healthy_sequences"
                    }
                ],
                "implanting": [{
                    "signals": ["signal1"],
                    "repertoires": 100,
                    "sequences": 10
                }]
            }
        })

        keys = ["repertoire_count", "sequence_count", "ml_methods", "simulation", "signals"]

        print(parsed)

        self.assertTrue(all([key in parsed.keys() for key in keys]))
        self.assertTrue(all([isinstance(item, MLMethod) for item in parsed["ml_methods"]]))
        self.assertTrue(all([all(isinstance(item, Signal) for item in item2["signals"]) for item2 in parsed["simulation"]]))

    def test_parse2(self):
        parsed = Parser.parse({
            "repertoire_count": 200,
            "sequence_count": 50,
            "ml_methods": {
                "LogisticRegression": {
                    "max_iter": [1000, 2000],
                    "penalty": ["l1", "l2"]
                },
                "SVM": {
                    "max_iter": [1000, 2000],
                    "penalty": ["l1", "l2"]
                }
            },
            "encoder": "KmerFrequencyEncoder",
            "encoder_params": {
                "normalization_type": "relative_frequency",
                "reads": "unique",
                "sequence_encoding": "identity",
                "k": 3
            },
            "simulation": {
                "motifs": [
                    {
                        "id": "motif1",
                        "seed": "CAS",
                        "instantiation": "identity"
                    }
                ],
                "signals": [
                    {
                        "id": "signal1",
                        "motifs": ["motif1"],
                        "implanting": "healthy_sequences"
                    }
                ],
                "implanting": [{
                    "signals": ["signal1"],
                    "repertoires": 100,
                    "sequences": 10
                }]
            }
        })

        keys = ["repertoire_count", "sequence_count", "ml_methods", "simulation", "signals"]

        self.assertTrue(all([key in parsed.keys() for key in keys]))
        self.assertTrue(all([isinstance(item, MLMethod) for item in parsed["ml_methods"]]))
        self.assertTrue(parsed["ml_methods"][0]._parameter_grid is not None
                        and len(parsed["ml_methods"][0]._parameter_grid["max_iter"]) == 2)
        self.assertTrue(parsed["ml_methods"][1]._parameter_grid is not None
                        and len(parsed["ml_methods"][1]._parameter_grid["max_iter"]) == 2)
        self.assertTrue(all([all(isinstance(item, Signal) for item in item2["signals"]) for item2 in parsed["simulation"]]))

    def test_parse_yaml_file(self):

        spec = {
                "repertoire_count": 200,
                "sequence_count": 50,
                "ml_methods": {
                    "LogisticRegression": {
                        "max_iter": 1000,
                        "penalty": ["l1", "l2"]
                    },
                    "SVM": {
                        "max_iter": [1000, 2000],
                        "penalty": ["l1", "l2"]
                    }
                },
                "encoder": "KmerFrequencyEncoder",
                "encoder_params": {
                    "normalization_type": "relative_frequency",
                    "reads": "unique",
                    "sequence_encoding": "identity",
                    "k": 3
                },
                "simulation": {
                    "motifs": [
                        {
                            "id": "motif1",
                            "seed": "CAS",
                            "instantiation": "identity"
                        }
                    ],
                    "signals": [
                        {
                            "id": "signal1",
                            "motifs": ["motif1"],
                            "implanting": "healthy_sequences"
                        }
                    ],
                    "implanting": [{
                        "signals": ["signal1"],
                        "repertoires": 100,
                        "sequences": 10
                    }]
                }
            }

        with open("./tmp_yaml_spec.yaml", "w") as file:
            yaml.dump(spec, file, default_flow_style=False)

        parsed = Parser.parse_yaml_file("./tmp_yaml_spec.yaml")

        keys = ["repertoire_count", "sequence_count", "ml_methods", "simulation", "signals"]

        self.assertTrue(all([key in parsed.keys() for key in keys]))
        self.assertTrue(all([isinstance(item, MLMethod) for item in parsed["ml_methods"]]))
        self.assertTrue(parsed["ml_methods"][0]._parameter_grid is not None
                        and len(parsed["ml_methods"][0]._parameter_grid["max_iter"]) == 1)
        self.assertTrue(parsed["ml_methods"][1]._parameter_grid is not None
                        and len(parsed["ml_methods"][1]._parameter_grid["max_iter"]) == 2)
        self.assertTrue(
            all([all(isinstance(item, Signal) for item in item2["signals"]) for item2 in parsed["simulation"]]))

        os.remove("./tmp_yaml_spec.yaml")
