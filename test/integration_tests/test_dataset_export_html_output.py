import shutil
from unittest import TestCase

import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestDatasetExportHTMLOutput(TestCase):

    def test_repertoire_dataset(self):

        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "integration_dataset_gen_html_repertoire/")
        dataset_path = path / "repertoire_dataset/"

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomRepertoireDataset",
                        "params": {
                            "repertoire_count": 10,
                            "sequence_count_probabilities": {
                                10: 1
                            },
                            "sequence_length_probabilities": {
                                12: 1
                            },
                            "labels": {"HLA": {"A": 0.5, "B": 0.5}},
                            "result_path": str(dataset_path)
                        }
                    }
                },
                "preprocessing_sequences": {
                    "p1": [
                        {
                            "my_filter": {
                                "ClonesPerRepertoireFilter": {
                                    "lower_limit": 1,
                                }
                            }
                        }
                    ]
                }
            },
            "instructions": {"instr1": {"type": "DatasetExport", "export_formats": ["Pickle", "AIRR"], "datasets": ["d1"], "preprocessing_sequence": "p1"}},
            "output": {"format": "HTML"}
        }

        specs_path = path / "specs.yaml"
        with open(specs_path, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_path, path / "result/")
        app.run()

        shutil.rmtree(path)


    def test_receptor_dataset(self):

        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "integration_dataset_gen_html_receptor/")
        dataset_path = path / "receptor_dataset/"

        specs = {
            "definitions": {
                "datasets": {
                    "receptordataset": {
                        "format": "RandomReceptorDataset",
                        "params": {
                            "receptor_count": 10,
                            "chain_1_length_probabilities": {
                                10: 1
                            },
                            "chain_2_length_probabilities": {
                                10: 1
                            },
                            "labels": {"epitope_1": {True: 0.5, False: 0.5},
                                       "epitope_2": {True: 0.5, False: 0.5}},
                            "result_path": str(dataset_path)
                        }
                    }
                }
            },
            "instructions": {"instr1": {"type": "DatasetExport", "export_formats": ["Pickle", "AIRR"], "datasets": ["receptordataset"]}},
            "output": {"format": "HTML"}
        }

        specs_path = path / "specs.yaml"
        with open(specs_path, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_path, path / "result/")
        app.run()

        shutil.rmtree(path)


    def test_sequence_dataset(self):

        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "integration_dataset_gen_html_sequence/")
        dataset_path = path / "sequence_dataset/"

        specs = {
            "definitions": {
                "datasets": {
                    "sequencedataset": {
                        "format": "RandomSequenceDataset",
                        "params": {
                            "sequence_count": 10,
                            "length_probabilities": {
                                10: 1
                            },
                            "labels": {"epitope_a": {True: 0.5, False: 0.5},
                                       "epitope_b": {True: 0.5, False: 0.5}},
                            "result_path": str(dataset_path)
                        }
                    }
                }
            },
            "instructions": {"instr1": {"type": "DatasetExport", "export_formats": ["Pickle", "AIRR"], "datasets": ["sequencedataset"]}},
            "output": {"format": "HTML"}
        }

        specs_path = path / "specs.yaml"
        with open(specs_path, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_path, path / "result/")
        app.run()

        shutil.rmtree(path)