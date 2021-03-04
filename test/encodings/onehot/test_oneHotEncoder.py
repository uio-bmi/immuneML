import os
import shutil
import unittest

import numpy as np

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestOneHotEncoder(unittest.TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _construct_test_repertoiredataset(self, path, positional):

        if positional:
            receptors1 = [ReceptorSequence("AAAAAAAAAAAAAAAAA", nucleotide_sequence="AAAAAAAAAAAAAAAAA", identifier="1"),
                          ReceptorSequence("AAAAAAAAAAAAAAAAA", nucleotide_sequence="AAAAAAAAAAAAAAAAA", identifier="1")]
            receptors2 = [ReceptorSequence("TTTTTTTTTTTTT", nucleotide_sequence='TTTTTTTTTTTTT', identifier="1")]
        else:
            receptors1 = [ReceptorSequence("AAAA", nucleotide_sequence="AAAA", identifier="1"),
                          ReceptorSequence("ATA", nucleotide_sequence="ATA", identifier="2"),
                          ReceptorSequence("ATA", nucleotide_sequence="ATA", identifier='3')]
            receptors2 = [ReceptorSequence("ATA", nucleotide_sequence="ATA", identifier="1"),
                          ReceptorSequence("TAA", nucleotide_sequence="TAA", identifier="2")]

        rep1 = Repertoire.build_from_sequence_objects(receptors1, metadata={"l1": 1, "l2": 2, "subject_id": "1"}, path=path)
        rep2 = Repertoire.build_from_sequence_objects(receptors2, metadata={"l1": 0, "l2": 3, "subject_id": "2"}, path=path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [0, 3])

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        return dataset, lc

    def test_not_positional(self):

        path = EnvironmentSettings.root_path / "test/tmp/onehot_vanilla/"

        PathBuilder.build(path)

        dataset, lc = self._construct_test_repertoiredataset(path, positional=False)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False, 'sequence_type': 'amino_acid',
                                                         "distance_to_seq_middle": 6, "flatten": False})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=lc,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        self.assertTrue(isinstance(encoded_data, RepertoireDataset))

        onehot_a = [1] + [0] * 19
        onehot_t = [0] * 16 + [1] + [0] * 3
        onehot_empty = [0] * 20

        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[0][0]], [onehot_a for i in range(4)])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[0][1]],
                             [onehot_a, onehot_t, onehot_a, onehot_empty])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[0][2]],
                             [onehot_a, onehot_t, onehot_a, onehot_empty])

        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[1][0]],
                             [onehot_a, onehot_t, onehot_a, onehot_empty])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[1][1]],
                             [onehot_t, onehot_a, onehot_a, onehot_empty])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[1][2]], [onehot_empty for i in range(4)])

        self.assertListEqual(list(encoded_data.encoded_data.example_ids), [repertoire.identifier for repertoire in dataset.get_data()])
        self.assertDictEqual(encoded_data.encoded_data.labels,
                             {"l1": [repertoire.metadata["l1"] for repertoire in dataset.get_data()],
                              "l2": [repertoire.metadata["l2"] for repertoire in dataset.get_data()]})

        shutil.rmtree(path)

    def test_positional(self):

        path = EnvironmentSettings.root_path / "test/tmp/onehot_positional/"

        PathBuilder.build(path)

        dataset, lc = self._construct_test_repertoiredataset(path, positional=True)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": True, 'sequence_type': 'amino_acid',
                                                         "distance_to_seq_middle": 6, "flatten": False})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=lc,
            pool_size=1,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        self.assertTrue(isinstance(encoded_data, RepertoireDataset))

        onehot_a = [1.0] + [0.0] * 19
        onehot_t = [0.0] * 16 + [1.0] + [0] * 3
        onehot_empty = [0.0] * 20

        # testing onehot dimensions, all but the last 3 value (last 3 = positional info)
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[0, 0, :, :-3]], [onehot_a for i in range(17)])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[0, 1, :, :-3]], [onehot_a for i in range(17)])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[1, 0, :, :-3]],
                             [onehot_t for i in range(13)] + [onehot_empty for i in range(4)])

        # testing positional dimensions
        precision = 5

        # Dimension: middle
        self.assertListEqual([round(val, precision) for val in encoded_data.encoded_data.examples[0, 0, :, 21]],
                             [round(val, precision) for val in
                              [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1, 1, 1, 1, 1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6, 0]])
        self.assertListEqual([round(val, precision) for val in encoded_data.encoded_data.examples[0, 1, :, 21]],
                             [round(val, precision) for val in
                              [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1, 1, 1, 1, 1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6, 0]])
        self.assertListEqual([round(val, precision) for val in encoded_data.encoded_data.examples[1, 0, :, 21]],
                             [round(val, precision) for val in
                              [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6, 0, 0, 0, 0, 0]])

        # Dimension: start
        self.assertListEqual([round(val, precision) for val in encoded_data.encoded_data.examples[0, 0, :, 20]],
                             [round(val, precision) for val in [1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertListEqual([round(val, precision) for val in encoded_data.encoded_data.examples[0, 1, :, 20]],
                             [round(val, precision) for val in [1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertListEqual([round(val, precision) for val in encoded_data.encoded_data.examples[1, 0, :, 20]],
                             [round(val, precision) for val in [1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # Dimension: end
        self.assertListEqual([round(val, precision) for val in encoded_data.encoded_data.examples[0, 0, :, 22]],
                             [round(val, precision) for val in [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1]])
        self.assertListEqual([round(val, precision) for val in encoded_data.encoded_data.examples[0, 1, :, 22]],
                             [round(val, precision) for val in [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1]])
        self.assertListEqual([round(val, precision) for val in encoded_data.encoded_data.examples[1, 0, :, 22]],
                             [round(val, precision) for val in [0, 0, 0, 0, 0, 0, 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1, 0, 0, 0, 0]])
        shutil.rmtree(path)

    def test_imgt_weights(self):
        path = EnvironmentSettings.root_path / "test/tmp/onehot_imgt/"

        PathBuilder.build(path)

        encoder = OneHotEncoder.build_object(RepertoireDataset(), **{"use_positional_info": True, 'sequence_type': 'amino_acid',
                                                                     "distance_to_seq_middle": 4,
                                                                     "flatten": False})

        # testing positional information for 'middle' (idx = 0) section
        self.assertListEqual(list(encoder._get_imgt_position_weights(9)[1]), [0, 1 / 4, 2 / 4, 3 / 4, 1, 3 / 4, 2 / 4, 1 / 4, 0])
        self.assertListEqual(list(encoder._get_imgt_position_weights(8)[1]), [0, 1 / 4, 2 / 4, 3 / 4, 3 / 4, 2 / 4, 1 / 4, 0])
        self.assertListEqual(list(encoder._get_imgt_position_weights(6)[1]), [0, 1 / 4, 2 / 4, 2 / 4, 1 / 4, 0])
        self.assertListEqual(list(encoder._get_imgt_position_weights(5)[1]), [0, 1 / 4, 2 / 4, 1 / 4, 0])

        # testing positional information for 'start' (idx = 1) and 'end' (idx = 2) section
        self.assertListEqual(list(encoder._get_imgt_position_weights(6, 8)[0]), [1, 3 / 4, 2 / 4, 1 / 4, 0, 0, 0, 0])
        self.assertListEqual(list(encoder._get_imgt_position_weights(6, 8)[2]), [0, 0, 1 / 4, 2 / 4, 3 / 4, 1, 0, 0])

        self.assertListEqual(list(encoder._get_imgt_position_weights(3)[0]), [1, 3 / 4, 2 / 4])
        self.assertListEqual(list(encoder._get_imgt_position_weights(3)[2]), [2 / 4, 3 / 4, 1])
        self.assertListEqual(list(encoder._get_imgt_position_weights(3, 6)[2]), [2 / 4, 3 / 4, 1, 0, 0, 0])

        shutil.rmtree(path)

    def test_nucleotide(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "onehot_nt/")

        dataset, lc = self._construct_test_repertoiredataset(path, positional=False)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False, "distance_to_seq_middle": None,
                                                         "flatten": False, 'sequence_type': 'nucleotide'})

        encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=path, label_config=lc, pool_size=1, learn_model=True, model={},
                                                                filename="dataset.pkl"))

        self.assertTrue(isinstance(encoded_dataset, RepertoireDataset))
        self.assertEqual((2, 3, 4, 4), encoded_dataset.encoded_data.examples.shape)
        self.assertTrue(np.array_equal(np.array([[[[1., 0., 0., 0.],   [1., 0., 0., 0.],   [1., 0., 0., 0.],   [1., 0., 0., 0.]],   # rep1: AAAA
                                                  [[1., 0., 0., 0.],   [0., 0., 0., 1.],   [1., 0., 0., 0.],   [0., 0., 0., 0.]],   # rep1: ATA
                                                  [[1., 0., 0., 0.],   [0., 0., 0., 1.],   [1., 0., 0., 0.],   [0., 0., 0., 0.]]],  # rep1: ATA
                                                 [[[1., 0., 0., 0.],   [0., 0., 0., 1.],   [1., 0., 0., 0.],   [0., 0., 0., 0.]],   # rep2: ATA
                                                  [[0., 0., 0., 1.],   [1., 0., 0., 0.],   [1., 0., 0., 0.],   [0., 0., 0., 0.]],   # rep2: TAA
                                                  [[0., 0., 0., 0.],  [0., 0., 0., 0.],   [0., 0., 0., 0.],   [0., 0., 0., 0.]]]]), # rep2: padding
                                       encoded_dataset.encoded_data.examples))

        shutil.rmtree(path)

    def test_repertoire_flattened(self):
        path = EnvironmentSettings.tmp_test_path / "onehot_recep_flat/"

        PathBuilder.build(path)

        dataset, lc = self._construct_test_repertoiredataset(path, positional=False)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False, "distance_to_seq_middle": None,
                                                         "flatten": True, 'sequence_type': 'amino_acid'})

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=lc,
            pool_size=1,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        self.assertTrue(isinstance(encoded_dataset, RepertoireDataset))

        onehot_a = [1.0] + [0.0] * 19
        onehot_t = [0.0] * 16 + [1.0] + [0] * 3
        onehot_empty = [0] * 20

        self.assertListEqual(list(encoded_dataset.encoded_data.examples[0]),
                             onehot_a + onehot_a + onehot_a + onehot_a + onehot_a + onehot_t + onehot_a + onehot_empty + onehot_a + onehot_t + onehot_a + onehot_empty)
        self.assertListEqual(list(encoded_dataset.encoded_data.examples[1]),
                             onehot_a + onehot_t + onehot_a + onehot_empty + onehot_t + onehot_a + onehot_a + onehot_empty + onehot_empty + onehot_empty + onehot_empty + onehot_empty)

        self.assertListEqual(list(encoded_dataset.encoded_data.feature_names),
                             [f"{seq}_{pos}_{char}" for seq in range(3) for pos in range(4) for char in EnvironmentSettings.get_sequence_alphabet()])

        shutil.rmtree(path)
