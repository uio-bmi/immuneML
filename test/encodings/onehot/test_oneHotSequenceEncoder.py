import shutil
from unittest import TestCase

from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestOneHotSequenceEncoder(TestCase):

    def _construct_test_dataset(self, path):
        sequences = [
            ReceptorSequence(amino_acid_sequence="AAAA", identifier="1", metadata=SequenceMetadata(custom_params={"l1": 1, "l2": 1})),
            ReceptorSequence(amino_acid_sequence="ATA", identifier="2", metadata=SequenceMetadata(custom_params={"l1": 2, "l2": 1})),
            ReceptorSequence(amino_acid_sequence="ATT", identifier="3", metadata=SequenceMetadata(custom_params={"l1": 1, "l2": 2}))]

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [1, 2])

        dataset = SequenceDataset.build_from_objects(sequences=sequences, file_size=10, path=path)

        return dataset, lc

    def test(self):
        path = EnvironmentSettings.tmp_test_path / "onehot_sequence/"
        PathBuilder.build(path)

        dataset, lc = self._construct_test_dataset(path)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False, 'sequence_type': 'amino_acid',
                                                         "distance_to_seq_middle": None, "flatten": False})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded/",
            label_config=lc,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        self.assertTrue(isinstance(encoded_data, SequenceDataset))

        onehot_a = [1] + [0] * 19
        onehot_t = [0] * 16 + [1] + [0] * 3
        onehot_empty = [0] * 20

        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[0]], [onehot_a for i in range(4)])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[1]], [onehot_a, onehot_t, onehot_a, onehot_empty])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[2]], [onehot_a, onehot_t, onehot_t, onehot_empty])

        self.assertListEqual(encoded_data.encoded_data.example_ids, [receptor.identifier for receptor in dataset.get_data()])
        self.assertDictEqual(encoded_data.encoded_data.labels,
                             {"l1": [receptor_seq.get_attribute("l1") for receptor_seq in dataset.get_data()],
                              "l2": [receptor_seq.get_attribute("l2") for receptor_seq in dataset.get_data()]})

        shutil.rmtree(path)

    def construct_test_flatten_dataset(self, path):
        sequences = [ReceptorSequence(amino_acid_sequence="AAATTT", identifier="1", metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="ATATAT", identifier="2", metadata=SequenceMetadata(custom_params={"l1": 2}))]

        PathBuilder.build(path)

        return SequenceDataset.build_from_objects(sequences=sequences, file_size=10, path=path)


    def test_sequence_flattened(self):
        path = EnvironmentSettings.root_path / "test/tmp/onehot_seq_flat/"

        PathBuilder.build(path)

        dataset = self.construct_test_flatten_dataset(path)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False, "distance_to_seq_middle": None, "flatten": True,
                                                         'sequence_type': 'amino_acid'})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=LabelConfiguration([Label(name="l1", values=[1, 0], positive_class="1")]),
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

        self.assertListEqual(list(encoded_data.encoded_data.feature_names), [f"{pos}_{char}" for pos in range(6) for char in EnvironmentSettings.get_sequence_alphabet()])
        shutil.rmtree(path)



