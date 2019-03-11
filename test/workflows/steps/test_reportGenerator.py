import os
import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.ReportGenerator import ReportGenerator


class TestReportGenerator(TestCase):
    def test_perform_step(self):
        path = EnvironmentSettings.root_path + "test/tmp/steps/"

        PathBuilder.build(path)

        rep1 = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA"),
                                     ReceptorSequence(amino_acid_sequence="AAAA"),
                                     ReceptorSequence(amino_acid_sequence="AAAAA"),
                                     ReceptorSequence(amino_acid_sequence="AAA")])
        rep2 = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAA"),
                                     ReceptorSequence(amino_acid_sequence="AAAA"),
                                     ReceptorSequence(amino_acid_sequence="AAAA"),
                                     ReceptorSequence(amino_acid_sequence="AAA")])

        with open(path + "rep1.pkl", "wb") as file:
            pickle.dump(rep1, file)
        with open(path + "rep2.pkl", "wb") as file:
            pickle.dump(rep2, file)

        dataset = Dataset(filenames=[path + "rep1.pkl", path + "rep2.pkl"])

        ReportGenerator.perform_step({
            "dataset": dataset,
            "result_path": path + "report_generator/",
            "reports": {
                "SequenceLengthDistribution": {
                    "report": SequenceLengthDistribution(),
                    "params": {
                        "result_path": path
                    }
                }
            },
            "batch_size": 2
        })

        self.assertTrue(os.path.isfile(path + "sequence_length_distribution.png"))

        shutil.rmtree(path)

