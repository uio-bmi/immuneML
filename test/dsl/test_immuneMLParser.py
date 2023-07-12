import shutil
from unittest import TestCase

import yaml
from yaml import YAMLError

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestImmuneMLParser(TestCase):
    def test_parse_yaml_file(self):
        path = EnvironmentSettings.tmp_test_path / "parser/"
        reps, metadata = RepertoireBuilder.build([["AAA", "CCC"], ["TTTT"]], path, {"default": [1, 2]})
        dataset = RepertoireDataset(repertoires=reps, metadata_file=metadata, labels={"default": [1, 2]})
        ImmuneMLExporter.export(dataset, path)

        spec = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "ImmuneML",
                        "params": {
                            "path": str(path / f"{dataset.name}.yaml")
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

        PathBuilder.remove_old_and_build(path)

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
