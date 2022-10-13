import os
import shutil
import numpy as np
from unittest import TestCase


from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.encodings.motif_encoding.PositionalMotifParams import PositionalMotifParams
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.NumpyHelper import NumpyHelper
from immuneML.util.PathBuilder import PathBuilder


class TestNumpyHelper(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _prepare_dataset(self, path):
        sequences = [ReceptorSequence(amino_acid_sequence="AA", identifier="1",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="CC", identifier="2",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="AC", identifier="3",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="CA", identifier="4",
                                      metadata=SequenceMetadata(custom_params={"l1": 1}))]

        PathBuilder.build(path)
        return SequenceDataset.build_from_objects(sequences, 100, PathBuilder.build(path / 'data'), 'd2')

    def test_get_numpy_sequence_representation(self):
        path = EnvironmentSettings.tmp_test_path / "positional_motif_sequence_encoder/test_np/"
        dataset = self._prepare_dataset(path = path)
        output = NumpyHelper.get_numpy_sequence_representation(dataset)

        expected = np.asarray(['A' 'A', 'C' 'C', 'A' 'C', 'C' 'A']).view('U1').reshape(4, -1)

        self.assertEqual(output.shape, expected.shape)

        for i in range(len(output)):
            self.assertListEqual(list(output[i]), list(expected[i]))

            for j in range(len(output[i])):
                self.assertEqual(type(output[i][j]), type(expected[i][j]))

        shutil.rmtree(path)