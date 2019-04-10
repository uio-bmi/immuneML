import os
import pickle
import shutil
from unittest import TestCase

import yaml

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.app.ImmuneMLApp import ImmuneMLApp
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestImmuneMLApp(TestCase):

    def create_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/immunemlapp/"
        PathBuilder.build(path)

        rep1 = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA"),
                                     ReceptorSequence(amino_acid_sequence="AAAA"),
                                     ReceptorSequence(amino_acid_sequence="AAAAA"),
                                     ReceptorSequence(amino_acid_sequence="AAA")],
                          metadata=RepertoireMetadata(custom_params={"CD": True}))
        rep2 = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA"),
                                     ReceptorSequence(amino_acid_sequence="AAAA"),
                                     ReceptorSequence(amino_acid_sequence="AAAA"),
                                     ReceptorSequence(amino_acid_sequence="AAA")],
                          metadata=RepertoireMetadata(custom_params={"CD": False}))

        with open(path + "rep1.pkl", "wb") as file:
            pickle.dump(rep1, file)
        with open(path + "rep2.pkl", "wb") as file:
            pickle.dump(rep2, file)
        with open(path + "rep3.pkl", "wb") as file:
            pickle.dump(rep1, file)
        with open(path + "rep4.pkl", "wb") as file:
            pickle.dump(rep2, file)
        with open(path + "rep5.pkl", "wb") as file:
            pickle.dump(rep1, file)
        with open(path + "rep6.pkl", "wb") as file:
            pickle.dump(rep2, file)

        dataset = Dataset(filenames=[path + "rep{}.pkl".format(i) for i in range(1, 7)], params={"CD": [True, False]})

        PickleExporter.export(dataset, path, "dataset.pkl")

        return path + "dataset.pkl"

    def test_run(self):

        dataset_path = self.create_dataset()

        specs = {
            "dataset_import": {
                "d1": {
                    "format": "Pickle",
                    "path": dataset_path,
                    "result_path": dataset_path
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
                    "model_selection_n_folds": -1
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
            }
        }

        path = EnvironmentSettings.root_path + "test/tmp/immunemlapp/"
        PathBuilder.build(path)
        specs_file = path + "specs.yaml"
        with open(specs_file, "w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_file, path)
        app.run()

        shutil.rmtree(os.path.dirname(dataset_path))
