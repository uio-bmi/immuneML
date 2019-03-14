import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.MetricType import MetricType
from source.ml_methods.LogisticRegression import LogisticRegression
from source.ml_methods.SVM import SVM
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.LeaveOneOutCV import LeaveOneOutCV


class TestLeaveOneOutCV(TestCase):
    def test_perform_step(self):

        path = EnvironmentSettings.root_path + "test/tmp/loocv/"
        PathBuilder.build(path)

        rep1 = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AACADAFGA"),
                                     ReceptorSequence(amino_acid_sequence="AACADAFGA"),
                                     ReceptorSequence(amino_acid_sequence="AACADAFGA")],
                          metadata=RepertoireMetadata(custom_params={"CD": True}))
        rep2 = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AACADAFGA"),
                                     ReceptorSequence(amino_acid_sequence="AACADAFGA"),
                                     ReceptorSequence(amino_acid_sequence="AACADAFGA")],
                          metadata=RepertoireMetadata(custom_params={"CD": False}))
        rep3 = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AACADAFGA"),
                                     ReceptorSequence(amino_acid_sequence="AACADAFGA"),
                                     ReceptorSequence(amino_acid_sequence="AACADAFGA")],
                          metadata=RepertoireMetadata(custom_params={"CD": True}))
        rep4 = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AACADAFGA"),
                                     ReceptorSequence(amino_acid_sequence="AACADAFGA"),
                                     ReceptorSequence(amino_acid_sequence="AACADAFGA")],
                          metadata=RepertoireMetadata(custom_params={"CD": False}))
        filenames = []
        filenames.append(path + "rep1.pkl")
        with open(filenames[-1], "wb") as file:
            pickle.dump(rep1, file)
        filenames.append(path + "rep2.pkl")
        with open(filenames[-1], "wb") as file:
            pickle.dump(rep2, file)
        filenames.append(path + "rep3.pkl")
        with open(filenames[-1], "wb") as file:
            pickle.dump(rep3, file)
        filenames.append(path + "rep4.pkl")
        with open(filenames[-1], "wb") as file:
            pickle.dump(rep4, file)

        dataset = Dataset(filenames=filenames, params={"CD": [True, False]})

        final_assessment = LeaveOneOutCV.run({
            "dataset": dataset,
            "result_path": path,
            "metrics": [MetricType.BALANCED_ACCURACY],
            "results_aggregator": "average",
            "methods": [LogisticRegression(), SVM()],
            "encoder": KmerFrequencyEncoder,
            "cv": 0,
            "encoder_params": {
                "model": {
                    "k": 2,
                    "normalization_type": NormalizationType.RELATIVE_FREQUENCY,
                    "reads": ReadsType.UNIQUE,
                    "sequence_encoding_strategy": SequenceEncodingType.CONTINUOUS_KMER
                }
            },
            "labels": ["CD"]
        })

        print(final_assessment)

        self.assertTrue("LogisticRegression" in final_assessment)
        self.assertTrue("SVM" in final_assessment)
        self.assertTrue("CD" in final_assessment["SVM"])
        self.assertTrue("BALANCED_ACCURACY" in final_assessment["SVM"]["CD"])

        shutil.rmtree(path)
