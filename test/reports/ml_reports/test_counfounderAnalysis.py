import os
import shutil
from unittest import TestCase

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression
from immuneML.reports.ml_reports.ConfounderAnalysis import ConfounderAnalysis
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator as RDG
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReadsType import ReadsType


class TestConfounderAnalysis(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_lr_model(self, path, encoded_data, label):
        # dummy logistic regression with 100 observations with 3 features belonging to 2 classes
        dummy_lr = LogisticRegression()
        dummy_lr.fit_by_cross_validation(encoded_data, optimization_metric="balanced_accuracy",
                                         number_of_splits=2, label=label)

        return dummy_lr

    def _make_dataset(self, path, size) -> RepertoireDataset:

        random_dataset = RDG.generate_repertoire_dataset(repertoire_count=size, sequence_count_probabilities={100: 1.},
                                                         sequence_length_probabilities={5: 1.},
                                                         labels={'disease': {True: 0.5, False: 0.5},
                                                                 'HLA': {True: 0.5, False: 0.5},
                                                                 'age': {True: 0.5, False: 0.5}}, path=path)

        return random_dataset

    def _encode_dataset(self, encoder, dataset, path, learn_model: bool = True):
        # encodes the repertoire by frequency of 3-mers
        lc = LabelConfiguration()
        lc.add_label("disease", [True, False])
        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=learn_model,
            model={}
        ))
        return encoded_dataset

    def _create_report(self, path):
        report = ConfounderAnalysis.build_object(metadata_labels=["age", "HLA"], name='test')

        report.label = Label("disease", [True, False])
        report.result_path = path
        encoder = KmerFrequencyEncoder.build_object(RepertoireDataset(), **{
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
            "reads": ReadsType.UNIQUE.name,
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
            "k": 3,
            'sequence_type': SequenceType.AMINO_ACID.name
        })
        report.train_dataset = self._encode_dataset(encoder, self._make_dataset(path / "train", size=100), path)
        report.test_dataset = self._encode_dataset(encoder, self._make_dataset(path / "test", size=40), path, learn_model=False)
        report.method = self._create_dummy_lr_model(path, report.train_dataset.encoded_data, Label("disease", [True, False]))

        return report

    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / "confounder_report/"
        PathBuilder.remove_old_and_build(path)

        report = self._create_report(path)

        # Running the report
        result = report._generate()

        shutil.rmtree(path)
