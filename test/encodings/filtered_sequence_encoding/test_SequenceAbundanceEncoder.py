import os
import shutil
from unittest import TestCase

import numpy as np

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.EncoderParams import EncoderParams
from source.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
from source.encodings.filtered_sequence_encoding.SequenceFilterHelper import SequenceFilterHelper
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.pairwise_repertoire_comparison.ComparisonDataBatch import ComparisonDataBatch
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestEmersonSequenceAbundanceEncoder(TestCase):
    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):
        path = EnvironmentSettings.tmp_test_path + "abundance_encoder/"
        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build([["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"]],
                                                        labels={"l1": [True, True, False, False]}, path=path)

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, identifier="1")

        encoder = SequenceAbundanceEncoder.build_object(dataset, **{
            "comparison_attributes": ["sequence_aas"],
            "p_value_threshold": 0.4, "sequence_batch_size": 4
        })

        label_config = LabelConfiguration([Label("l1", [True, False])])

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_configuration=label_config))

        self.assertTrue(np.array_equal(np.array([[1, 4], [1, 6], [1, 3], [1, 6]]), encoded_dataset.encoded_data.examples))

        encoder.p_value_threshold = 0.05

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_configuration=label_config))

        self.assertTrue(np.array_equal(np.array([[0, 4], [0, 6], [0, 3], [0, 6]]), encoded_dataset.encoded_data.examples))

        shutil.rmtree(path)

    def test__build_abundance_matrix(self):
        path = EnvironmentSettings.tmp_test_path + "abundance_encoder_matrix/"
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
            "p_value_threshold": 0.4, "sequence_batch_size": 4
        })

        abundance_matrix = encoder._build_abundance_matrix(comparison_data, ["rep_0", "rep_1", "rep_2", "rep_3"], sequence_p_value_indices)

        self.assertTrue(np.array_equal(expected_abundance_matrix, abundance_matrix))

        shutil.rmtree(path)

    def test_find_label_associated_sequence_p_values(self):
        path = EnvironmentSettings.tmp_test_path + "comparison_data_find_label_assocseqpvalues/"
        PathBuilder.build(path)

        repertoires = [Repertoire.build_from_sequence_objects([ReceptorSequence()], path, {
            "l1": val, "subject_id": subject_id
        }) for val, subject_id in zip([True, True, False, False], ["rep_0", "rep_1", "rep_2", "rep_3"])]

        col_name_index = {repertoires[index].identifier: index for index in range(len(repertoires))}

        comparison_data = ComparisonData(repertoire_ids=[repertoire.identifier for repertoire in repertoires],
                                         comparison_attributes=["sequence_aas"], sequence_batch_size=4, path=path)
        comparison_data.batches = [ComparisonDataBatch(**{'matrix': np.array([[1., 0., 0., 0.],
                                                                              [1., 1., 0., 0.]]),
                                                          'items': [('GGG',), ('III',)],
                                                          'repertoire_index_mapping': col_name_index, 'path': path, 'identifier': 0}),
                                   ComparisonDataBatch(**{'matrix': np.array([[1., 1., 0., 1.],
                                                                              [1., 1., 1., 1.]]),
                                                          'items': [('LLL',), ('MMM',)],
                                                          'repertoire_index_mapping': col_name_index, 'path': path, 'identifier': 1}),
                                   ComparisonDataBatch(**{'matrix': np.array([[0., 1., 0., 0.],
                                                                              [0., 1., 0., 1.]]),
                                                          'items': [('DDD',), ('EEE',)],
                                                          'repertoire_index_mapping': col_name_index, 'path': path, 'identifier': 2}),
                                   ComparisonDataBatch(**{'matrix': np.array([[0., 1., 1., 1.],
                                                                              [0., 0., 1., 1.]]),
                                                          'items': [('FFF',), ('CCC',)],
                                                          'repertoire_index_mapping': col_name_index, 'path': path, 'identifier': 3}),
                                   ComparisonDataBatch(**{'matrix': np.array([[0., 0., 0., 1.]]),
                                                          'items': [('AAA',)],
                                                          'repertoire_index_mapping': col_name_index, 'path': path, 'identifier': 4})]

        p_values = SequenceFilterHelper.find_label_associated_sequence_p_values(comparison_data, repertoires, "l1", [True, False])

        self.assertTrue(
            np.allclose([2, 0.3333333333333334, 1., 1., 2, 1., 1., 0.3333333333333334, 2], p_values, equal_nan=True))

        shutil.rmtree(path)
