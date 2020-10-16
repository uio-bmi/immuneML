import os
import shutil
import unittest

from source.IO.dataset_import.GenericImport import GenericImport
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.EncoderParams import EncoderParams
from source.encodings.onehot.OneHotEncoder import OneHotEncoder
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder


class TestOneHotEncoder(unittest.TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _construct_test_dataset(self, path, positional):
        receptors1 = ReceptorSequenceList()
        receptors2 = ReceptorSequenceList()

        if positional:
            [receptors1.append(seq) for seq in
             [ReceptorSequence("AAAAAAAAAAAAAAAAA", identifier="1"), ReceptorSequence("AAAAAAAAAAAAAAAAA", identifier="1")]]
            [receptors2.append(seq) for seq in [ReceptorSequence("TTTTTTTTTTTTT", identifier="1")]]
        else:
            [receptors1.append(seq) for seq in
             [ReceptorSequence("AAAA", identifier="1"), ReceptorSequence("ATA", identifier="2"), ReceptorSequence("ATA", identifier='3')]]
            [receptors2.append(seq) for seq in [ReceptorSequence("ATA", identifier="1"), ReceptorSequence("TAA", identifier="2")]]

        rep1 = Repertoire.build_from_sequence_objects(receptors1,
                                                      metadata={"l1": 1, "l2": 2, "subject_id": "1"}, path=path)

        rep2 = Repertoire.build_from_sequence_objects(receptors2,
                                                      metadata={"l1": 0, "l2": 3, "subject_id": "2"}, path=path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [0, 3])

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        return dataset, lc

    def test_not_positional(self):

        path = EnvironmentSettings.root_path + "test/tmp/onehot_vanilla/"

        PathBuilder.build(path)

        dataset, lc = self._construct_test_dataset(path, positional=False)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False,
                                                         "distance_to_seq_middle": 6,
                                                         "flatten": False})

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

        path = EnvironmentSettings.root_path + "test/tmp/onehot_positional/"

        PathBuilder.build(path)

        dataset, lc = self._construct_test_dataset(path, positional=True)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": True,
                                                         "distance_to_seq_middle": 6,
                                                         "flatten": False})

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
        path = EnvironmentSettings.root_path + "test/tmp/onehot_imgt/"

        PathBuilder.build(path)

        encoder = OneHotEncoder.build_object(RepertoireDataset(), **{"use_positional_info": True,
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


    def test_sequence_flattened(self):
        path = EnvironmentSettings.root_path + "test/tmp/onehot_imgt/"

        PathBuilder.build(path)

        file_content = """sequence_aas	is_binding
AAATTT	1
ATATAT	0"""
        path = EnvironmentSettings.root_path + "test/tmp/iovdjdb/"
        PathBuilder.build(path)

        with open(path + "receptors.tsv", "w") as file:
            file.writelines(file_content)

        dataset = GenericImport.import_dataset({"is_repertoire": False, "result_path": path, "paired": False, "path": path, "sequence_file_size": 1,
                                              "column_mapping": None, "metadata_column_mapping": {"is_binding": "is_binding"},
                                              "separator": "\t", "region_type": "FULL_SEQUENCE"}, "mini_mason_dataset")

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False, "distance_to_seq_middle": None, "flatten": True})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=LabelConfiguration([Label(name="is_binding", values=["1", "0"], positive_class="1")]),
            pool_size=1,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        self.assertTrue(isinstance(encoded_data, SequenceDataset))

        onehot_a = [1.0] + [0.0] * 19
        onehot_t = [0.0] * 16 + [1.0] + [0] * 3

        self.assertListEqual(list(encoded_data.encoded_data.examples[0]), onehot_a+onehot_a+onehot_a+onehot_t+onehot_t+onehot_t)
        self.assertListEqual(list(encoded_data.encoded_data.examples[1]), onehot_a+onehot_t+onehot_a+onehot_t+onehot_a+onehot_t)

        shutil.rmtree(path)  # todo add test flattened + positional?

