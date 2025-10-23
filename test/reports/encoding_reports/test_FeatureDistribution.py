import os
import random
import shutil
import string
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy import sparse

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.EncodedData import EncodedData
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFreqSequenceEncoder import KmerFreqSequenceEncoder
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.FeatureDistribution import FeatureDistribution
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReadsType import ReadsType


class TestFeatureDistribution(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_encoded_rep_data(self, path):
        n_subjects = 50
        n_features = 30

        kmers = [''.join(random.choices(string.ascii_uppercase, k=3)) for i in range(n_features)]

        encoded_data = {
            'examples': sparse.csr_matrix(
                np.random.normal(50, 10, n_subjects * n_features).reshape((n_subjects, n_features))),
            'example_ids': [''.join(random.choices(string.ascii_uppercase, k=4)) for i in range(n_subjects)],
            'labels': {"l1": [i % 2 for i in range(n_subjects)]
            },
            'feature_names': kmers,
            'feature_annotations': pd.DataFrame({
                "sequence": kmers
            }),
            'encoding': "random"
        }

        metadata_filepath = path / "metadata.csv"

        metadata = pd.DataFrame({"patient": np.array([i for i in range(n_subjects)]),
                                 "l1": encoded_data["labels"]["l1"]})

        metadata.to_csv(metadata_filepath, index=False)

        dataset = RepertoireDataset(encoded_data=EncodedData(**encoded_data), metadata_file=metadata_filepath)

        return dataset

    def _create_dummy_encoded_seq_data(self, path):
        sequences = [ReceptorSequence(sequence_aa="GGGCCC", sequence="GGGCCC", sequence_id="1",
                                      metadata={"l1": 1}),
                     ReceptorSequence(sequence_aa="ACACAC", sequence="ACACAC", sequence_id="2",
                                      metadata={"l1": 2}),
                     ReceptorSequence(sequence_aa="CCCAAA", sequence="CCCAAA", sequence_id="3",
                                      metadata={"l1": 1}),
                     ReceptorSequence(sequence_aa="AAACCC", sequence="AAACCC", sequence_id="4",
                                      metadata={"l1": 2}),
                     ReceptorSequence(sequence_aa="ACACAC", sequence="ACACAC", sequence_id="5",
                                      metadata={"l1": 1}),
                     ReceptorSequence(sequence_aa="CCCAAA", sequence="CCCAAA", sequence_id="6",
                                      metadata={"l1": 2}),
                     ReceptorSequence(sequence_aa="AAACCC", sequence="AAACCC", sequence_id="7",
                                      metadata={"l1": 1}),
                     ReceptorSequence(sequence_aa="ACACAC", sequence="ACACAC", sequence_id="8",
                                      metadata={"l1": 2}),
                     ReceptorSequence(sequence_aa="CCCAAA", sequence="CCCAAA", sequence_id="9",
                                      metadata={"l1": 1})]

        dataset = SequenceDataset.build_from_objects(sequences, PathBuilder.build(path / 'data'), 'seq_data',
                                                     labels={"l1": [1, 2]})

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        encoder = KmerFreqSequenceEncoder.build_object(dataset, **{
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
                "reads": ReadsType.UNIQUE.name,
                "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
                "sequence_type": SequenceType.NUCLEOTIDE.name,
                "k": 3
            })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded_seq_data",
            label_config=lc,
            pool_size=2,
            learn_model=True,
            model={},
        ))

        return encoded_dataset

    def test_generate_report_rep_data(self):
        path = EnvironmentSettings.tmp_test_path / "featuredistribution_rep_dataset/"
        PathBuilder.remove_old_and_build(path)

        dataset = self._create_dummy_encoded_rep_data(path)

        report = FeatureDistribution.build_object(**{"dataset": dataset,
                                                     "result_path": path,
                                                     "mode": "sparse", "error_function": "sem",
                                                     "color_grouping_label": "l1",
                                                     "plot_all_features": True,
                                                     "plot_top_n": 10,
                                                     "plot_bottom_n": 5})

        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertEqual(result.output_figures[0].path, path / "feature_distributions_all.html")
        self.assertEqual(result.output_tables[0].path, path / "feature_distributions_all.csv")

        content = pd.read_csv(path / "feature_values.csv")
        self.assertListEqual(list(content.columns),
                             ["patient", "l1", "example_id", "sequence", "feature", "value"])

        # report should succeed to build_from_objects but check_prerequisites should be false when data is not encoded
        report = FeatureDistribution.build_object(**{"dataset": RepertoireDataset(),
                                                     "result_path": path})

        self.assertFalse(report.check_prerequisites())

        shutil.rmtree(path)

    def test_generate_report_seq_data(self):
        path = EnvironmentSettings.tmp_test_path / "featuredistribution_seq_dataset/"
        PathBuilder.remove_old_and_build(path)

        dataset = self._create_dummy_encoded_seq_data(path)

        report = FeatureDistribution.build_object(**{"dataset": dataset,
                                                     "result_path": path,
                                                     "mode": "sparse",
                                                     "color_grouping_label": "l1",
                                                     "plot_all_features": True,
                                                     "plot_top_n": 4,
                                                     "plot_bottom_n": 5})

        report.check_prerequisites()

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertEqual(result.output_figures[0].path, path / "feature_distributions_all.html")
        self.assertEqual(result.output_tables[0].path, path / "feature_distributions_all.csv")

        content = pd.read_csv(path / "feature_values.csv")
        self.assertListEqual(list(content.columns),
                             ["l1", "example_id", "feature", "value"])

        report = FeatureDistribution.build_object(**{"dataset": SequenceDataset(),
                                                     "result_path": path})

        self.assertFalse(report.check_prerequisites())

        shutil.rmtree(path)
