import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from immuneML.pairwise_repertoire_comparison.ComparisonDataBatch import ComparisonDataBatch
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestEmersonSequenceAbundanceEncoder(TestCase):
    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):
        path = EnvironmentSettings.tmp_test_path / "abundance_encoder/"
        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build([["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"]],
                                                        labels={"l1": [True, True, False, False]}, path=path)

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, identifier="1")

        encoder = SequenceAbundanceEncoder.build_object(dataset, **{
            "comparison_attributes": ["sequence_aas"],
            "p_value_threshold": 0.4, "sequence_batch_size": 4, "repertoire_batch_size": 8
        })

        label_config = LabelConfiguration([Label("l1", [True, False], positive_class=True)])

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_config=label_config))

        self.assertTrue(np.array_equal(np.array([[1, 4], [1, 6], [0, 3], [0, 6]]), encoded_dataset.encoded_data.examples))

        encoder.p_value_threshold = 0.05

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_config=label_config))

        self.assertTrue(np.array_equal(np.array([[0, 4], [0, 6], [0, 3], [0, 6]]), encoded_dataset.encoded_data.examples))

        shutil.rmtree(path)

    def test__build_abundance_matrix(self):
        path = EnvironmentSettings.tmp_test_path / "abundance_encoder_matrix/"
        PathBuilder.build(path)
        expected_abundance_matrix = np.array([[1, 4], [1, 6], [1, 3], [1, 6]])

        comparison_data = ComparisonData(repertoire_ids=["rep_0", "rep_1", "rep_2", "rep_3"],
                                         comparison_attributes=["sequence_aas"], sequence_batch_size=2, path=path)
        comparison_data.batches = [ComparisonDataBatch(matrix=np.array([[1., 0., 0., 0.],
                                                                        [1., 1., 0., 0.]]),
                                                       items=[('GGG',), ('III',)], identifier=0,
                                                       repertoire_index_mapping={'rep_0': 0, 'rep_1': 1, 'rep_2': 2, 'rep_3': 3}, path=path),
                                   ComparisonDataBatch(matrix=np.array([[1., 1., 0., 1.],
                                                                        [1., 1., 1., 1.]]),
                                                       items=[('LLL',), ('MMM',)], identifier=1, path=path,
                                                       repertoire_index_mapping={'rep_0': 0, 'rep_1': 1, 'rep_2': 2, 'rep_3': 3}),
                                   ComparisonDataBatch(matrix=np.array([[0., 1., 0., 0.],
                                                                        [0., 1., 0., 1.]]),
                                                       items=[('DDD',), ('EEE',)], identifier=2, path=path,
                                                       repertoire_index_mapping={'rep_0': 0, 'rep_1': 1, 'rep_2': 2, 'rep_3': 3}),
                                   ComparisonDataBatch(matrix=np.array([[0., 1., 1., 1.],
                                                                        [0., 0., 1., 1.]]),
                                                       items=[('FFF',), ('CCC',)], identifier=3, path=path,
                                                       repertoire_index_mapping={'rep_0': 0, 'rep_1': 1, 'rep_2': 2, 'rep_3': 3}),
                                   ComparisonDataBatch(matrix=np.array([[0., 0., 0., 1.]]),
                                                       items=[('AAA',)], identifier=4, path=path,
                                                       repertoire_index_mapping={'rep_0': 0, 'rep_1': 1, 'rep_2': 2, 'rep_3': 3})]
        comparison_data.item_count = 9

        p_value = 0.4
        sequence_p_value_indices = np.array([1., 0.3333333333333334, 1., 1., 1., 1., 1., 0.3333333333333334, 1.]) < p_value

        encoder = SequenceAbundanceEncoder.build_object(RepertoireDataset(), **{
            "comparison_attributes": ["sequence_aas"],
            "p_value_threshold": 0.4, "sequence_batch_size": 4, "repertoire_batch_size": 10
        })

        abundance_matrix = encoder._build_abundance_matrix(comparison_data, ["rep_0", "rep_1", "rep_2", "rep_3"], sequence_p_value_indices)

        self.assertTrue(np.array_equal(expected_abundance_matrix, abundance_matrix))

        shutil.rmtree(path)
