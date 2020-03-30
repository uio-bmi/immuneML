import os
import shutil
from unittest import TestCase

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app import ImmuneMLApp
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestImmuneMLApp(TestCase):

    def create_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/immunemlapp/"
        PathBuilder.build(path)

        sequences1 = [ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A", count=2), identifier="1"),
                      ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B", count=4), identifier="2"),
                      ReceptorSequence(amino_acid_sequence="AAAAA", metadata=SequenceMetadata(chain="B", count=3), identifier="3"),
                      ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A", count=2), identifier="4")]
        sequences2 =[ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="B", count=2), identifier="5"),
                     ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="A", count=3), identifier="6"),
                     ReceptorSequence(amino_acid_sequence="AAAA", metadata=SequenceMetadata(chain="B", count=4), identifier="7"),
                     ReceptorSequence(amino_acid_sequence="AAA", metadata=SequenceMetadata(chain="A", count=2), identifier="8")]

        dataset = RepertoireDataset(repertoires=[Repertoire.build_from_sequence_objects(sequences1 if i % 2 == 0 else sequences2,
                                                                                        path,
                                                                                        metadata={"CD": True if i % 2 == 0 else False, "donor": f"rep{i%12*2}"})
                                                 for i in range(1, 14)],
                                    params={"CD": [True, False]})

        PickleExporter.export(dataset, path, "dataset.pkl")

        return path + "dataset.pkl"

    def test_run(self):

        dataset_path = self.create_dataset()

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "Pickle",
                        "params": {
                            "path": dataset_path,
                            "result_path": dataset_path
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
                "preprocessing_sequences": {
                    "seq1": [
                        {"collect": {
                            "PatientRepertoireCollector": {}
                        }},
                        {
                            "count_filter": {
                                "SequenceClonalCountFilter": {
                                    "remove_without_count": True,
                                    "low_count_limit": 3,
                                    "batch_size": 4
                                }
                            }
                        }
                    ]
                },
                "reports": {
                    "rep1": {
                        "SequenceLengthDistribution": {
                            "batch_size": 3
                        }
                    }
                },
            },
            "instructions": {
                "inst1": {
                    "type": "HPOptimization",
                    "settings": [
                        {
                            "preprocessing": "seq1",
                            "encoding": "e1",
                            "ml_method": "simpleLR"
                        },
                        {
                            "preprocessing": "seq1",
                            "encoding": "e2",
                            "ml_method": "simpleLR"
                        }
                    ],
                    "assessment": {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "reports": {
                            "data_splits": []
                        }
                    },
                    "selection": {
                        "reports": {
                            "data_splits": ["rep1"],
                            "models": [],
                            "optimal_models": []
                        }
                    },
                    "labels": ["CD"],
                    "dataset": "d1",
                    "strategy": "GridSearch",
                    "metrics": ["accuracy", "auc"],
                    "reports": ["rep1"],
                    "batch_size": 10,
                    "optimization_metric": "accuracy"
                }
            }
        }

        path = EnvironmentSettings.root_path + "test/tmp/immunemlapp/"
        PathBuilder.build(path)
        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp.ImmuneMLApp(specs_file, path)
        app.run()

        self.assertTrue(os.path.isfile(path+"full_specs.yaml"))
        with open(path+"full_specs.yaml", "r") as file:
            full_specs = yaml.load(file, Loader=yaml.FullLoader)

        self.assertTrue("split_strategy" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["split_strategy"] == "random")
        self.assertTrue("split_count" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["split_count"] == 1)
        self.assertTrue("training_percentage" in full_specs["instructions"]["inst1"]["selection"] and full_specs["instructions"]["inst1"]["selection"]["training_percentage"] == 0.7)

        with self.assertRaises(AssertionError):
            ImmuneMLApp.main([None, specs_file])

        shutil.rmtree(os.path.dirname(dataset_path))
