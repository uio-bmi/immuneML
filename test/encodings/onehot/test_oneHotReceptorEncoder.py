import shutil
from unittest import TestCase

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.receptor.TCABReceptor import TCABReceptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestOneHotSequenceEncoder(TestCase):

    def _construct_test_dataset(self, path, dataset_size: int = 50):
        receptors = [TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="AAAA"),
                                  beta=ReceptorSequence(amino_acid_sequence="ATA"),
                                  metadata={"l1": 1}, identifier=str("1")),
                     TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="ATA"),
                                  beta=ReceptorSequence(amino_acid_sequence="ATT"),
                                  metadata={"l1": 2}, identifier=str("2"))]

        PathBuilder.build(path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        dataset = ReceptorDataset.build(receptors, 2, path)
        return dataset, lc

    def test(self):
        path = EnvironmentSettings.tmp_test_path / "onehot_sequence_1/"
        PathBuilder.build(path)

        dataset, lc = self._construct_test_dataset(path)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False, 'sequence_type': 'amino_acid',
                                                         "distance_to_seq_middle": 6, "flatten": False})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        self.assertTrue(isinstance(encoded_data, ReceptorDataset))

        onehot_a = [1] + [0] * 19
        onehot_t = [0] * 16 + [1] + [0] * 3
        onehot_empty = [0] * 20

        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[0, 0]], [onehot_a for i in range(4)])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[0, 1]],
                             [onehot_a, onehot_t, onehot_a, onehot_empty])

        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[1, 0]],
                             [onehot_a, onehot_t, onehot_a, onehot_empty])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[1, 1]],
                             [onehot_a, onehot_t, onehot_t, onehot_empty])

        shutil.rmtree(path)

    def construct_test_flatten_dataset(self, path):
        receptors = [TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="AAATTT", identifier="1a"),
                                  beta=ReceptorSequence(amino_acid_sequence="ATATAT", identifier="1b"),
                                  metadata={"l1": 1},
                                  identifier="1"),
                     TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="AAAAAA", identifier="2a"),
                                  beta=ReceptorSequence(amino_acid_sequence="AAAAAA", identifier="2b"),
                                  metadata={"l1": 2},
                                  identifier="2"),
                     TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="AAAAAA", identifier="2a"),
                                  beta=ReceptorSequence(amino_acid_sequence="AAAAAA", identifier="2b"),
                                  metadata={"l1": 2},
                                  identifier="2")]

        return ReceptorDataset.build(receptors, 10, path)

    def test_receptor_flattened(self):
        path = EnvironmentSettings.root_path / "test/tmp/onehot_recep_flat/"

        PathBuilder.build(path)

        dataset = self.construct_test_flatten_dataset(path)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False, "distance_to_seq_middle": None,
                                                         'sequence_type': 'amino_acid', "flatten": True})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=LabelConfiguration([Label(name="l1", values=[1, 0], positive_class="1")]),
            pool_size=1,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        self.assertTrue(isinstance(encoded_data, ReceptorDataset))

        onehot_a = [1.0] + [0.0] * 19
        onehot_t = [0.0] * 16 + [1.0] + [0] * 3

        self.assertListEqual(list(encoded_data.encoded_data.examples[0]), onehot_a+onehot_a+onehot_a+onehot_t+onehot_t+onehot_t+onehot_a+onehot_t+onehot_a+onehot_t+onehot_a+onehot_t)
        self.assertListEqual(list(encoded_data.encoded_data.examples[1]), onehot_a*12)
        self.assertListEqual(list(encoded_data.encoded_data.examples[2]), onehot_a*12)

        self.assertListEqual(list(encoded_data.encoded_data.feature_names), [f"{chain}_{pos}_{char}" for chain in ("alpha","beta") for pos in range(6) for char in EnvironmentSettings.get_sequence_alphabet()])

        shutil.rmtree(path)
