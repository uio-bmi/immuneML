import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.EncoderParams import EncoderParams
from source.encodings.onehot.OneHotEncoder import OneHotEncoder
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder


class TestOneHotSequenceEncoder(TestCase):

    def _construct_test_dataset(self, path, dataset_size: int = 50):
        receptors = [TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="AAAA"),
                                  beta=ReceptorSequence(amino_acid_sequence="ATA"),
                                  metadata={"l1": 1}, identifier=str("1")),
                     TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="ATA"),
                                  beta=ReceptorSequence(amino_acid_sequence="ATT"),
                                  metadata={"l1": 2}, identifier=str("2"))]

        PathBuilder.build(path)
        filename = "{}receptors.pkl".format(path)
        with open(filename, "wb") as file:
            pickle.dump(receptors, file)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        dataset = ReceptorDataset(params={"l1": [1, 2]}, filenames=[filename], identifier="d1")
        return dataset, lc

    def test(self):
        path = EnvironmentSettings.tmp_test_path + "onehot_sequence_1/"
        PathBuilder.build(path)

        dataset, lc = self._construct_test_dataset(path)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False,
                                                         "distance_to_seq_middle": 6})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=f"{path}encoded/",
            label_configuration=lc,
            batch_size=2,
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
