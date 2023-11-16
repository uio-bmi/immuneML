import shutil

import pytest
import yaml
from yaml import YAMLError

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.LogisticRegression import LogisticRegression
from immuneML.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


def test_parse_iml_yaml_file():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "parser/")
    reps, metadata = RepertoireBuilder.build([["AAA", "CCC"], ["TTTT"]], path / 'tmp_data', {"default": [1, 2]})
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

            },
            "example_weightings": {
                "w1": {
                    "PredefinedWeighting":
                        {"file_path": "test"}
                }
            }
        },
        "instructions": {}
    }

    specs_filename = path / "tmp_yaml_spec.yaml"

    with specs_filename.open("w") as file:
        yaml.dump(spec, file, default_flow_style=False)

    symbol_table, _ = ImmuneMLParser.parse_yaml_file(specs_filename, result_path=path)

    assert all([symbol_table.contains(key) for key in ["simpleLR", "rep1", "a1", "d1"]])
    assert isinstance(symbol_table.get("d1"), RepertoireDataset)
    assert isinstance(symbol_table.get("rep1"), SequenceLengthDistribution)
    assert isinstance(symbol_table.get("simpleLR2"), LogisticRegression)

    with pytest.raises(YAMLError):
        with specs_filename.open("r") as file:
            specs_text = file.readlines()
        specs_text[0] = "        definitions:"
        with specs_filename.open("w") as file:
            file.writelines(specs_text)

        ImmuneMLParser.parse_yaml_file(specs_filename, result_path=path)

    shutil.rmtree(path)
