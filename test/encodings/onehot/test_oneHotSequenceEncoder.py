import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.encodings.EncoderParams import EncoderParams
from source.encodings.onehot.OneHotEncoder import OneHotEncoder
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder


class TestOneHotSequenceEncoder(TestCase):

    def _construct_test_dataset(self, path):
        sequences = [ReceptorSequence(amino_acid_sequence="AAAA", identifier="1", metadata=SequenceMetadata(custom_params={"l1": 1, "l2": 1})),
                     ReceptorSequence(amino_acid_sequence="ATA", identifier="2", metadata=SequenceMetadata(custom_params={"l1": 2, "l2": 1})),
                     ReceptorSequence(amino_acid_sequence="ATT", identifier="3", metadata=SequenceMetadata(custom_params={"l1": 1, "l2": 2}))]

        filename = "{}sequences.pkl".format(path)
        with open(filename, "wb") as file:
            pickle.dump(sequences, file)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [1, 2])

        dataset = SequenceDataset(params={"l1": [1, 2]}, filenames=[filename], identifier="d1")

        return dataset, lc


    def test(self):
        path = EnvironmentSettings.tmp_test_path + "onehot_sequence/"
        PathBuilder.build(path)

        dataset, lc = self._construct_test_dataset(path)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False,
                                                         "distance_to_seq_middle": 6})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_configuration=lc,
            batch_size=2,
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
        self.assertDictEqual(encoded_data.encoded_data.labels, {"l1": [receptor_seq.get_attribute("l1") for receptor_seq in dataset.get_data()],
                                                   "l2": [receptor_seq.get_attribute("l2") for receptor_seq in dataset.get_data()]})

        shutil.rmtree(path)
