from unittest import TestCase

from source.dsl_parsers.Parser import Parser
from source.ml_methods.MLMethod import MLMethod
from source.simulation.implants.Signal import Signal


class TestParser(TestCase):
    def test_parse(self):
        parsed = Parser.parse({
            "repertoire_count": 200,
            "sequence_count": 50,
            "ml_methods": ["LogisticRegression", "SVM", "RandomForest"],
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
