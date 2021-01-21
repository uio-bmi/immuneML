import shutil
from unittest import TestCase

import yaml
from yaml import YAMLError
from pathlib import Path

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.dsl.ImmuneMLParser import ImmuneMLParser
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestImmuneMLParser(TestCase):
    def test_parse_yaml_file(self):
        path = EnvironmentSettings.root_path / "test/tmp/parser/"
        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["AAA", "CCC"], ["TTTT"]], path, {"default": [1, 2]})[0],
                                    params={"default": [1, 2]})
        PickleExporter.export(dataset, path)

        spec = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "Pickle",
                        "params": {
                            "path": str(path / f"{dataset.name}.iml_dataset"),
                        }
                    }
                },
                "encodings": {
                    "a1": {
                        "Word2Vec": {
                            "k": 3,
                            "model_type": "sequence",
                            "vector_size": 8,
                        }
                    },
                    "a2": "Word2Vec"
                },
                "ml_methods": {
                    "simpleLR": {
                        "LogisticRegression":{
                            "penalty": "l1"
                        },
                        "model_selection_cv": False,
                        "model_selection_n_folds": -1,
                    },
                    "simpleLR2": "LogisticRegression"
                },
                "reports": {
                    "rep1": "SequenceLengthDistribution"

                }
            },
            "instructions": {}
        }

        PathBuilder.build(path)

        specs_filename = path / "tmp_yaml_spec.yaml"

        with specs_filename.open("w") as file:
            yaml.dump(spec, file, default_flow_style=False)

        symbol_table, _ = ImmuneMLParser.parse_yaml_file(specs_filename, result_path=path)

        self.assertTrue(all([symbol_table.contains(key) for key in
                             ["simpleLR", "rep1", "a1", "d1"]]))
        self.assertTrue(isinstance(symbol_table.get("d1"), RepertoireDataset))

        with self.assertRaises(YAMLError):
            with specs_filename.open("r") as file:
                specs_text = file.readlines()
            specs_text[0] = "        definitions:"
            with specs_filename.open("w") as file:
                file.writelines(specs_text)

            ImmuneMLParser.parse_yaml_file(specs_filename, result_path=path)

        shutil.rmtree(path)
