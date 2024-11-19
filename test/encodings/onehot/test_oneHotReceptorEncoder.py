import shutil
from unittest import TestCase

from immuneML.data_model.SequenceParams import ChainPair
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.data_model.SequenceSet import Receptor
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestOneHotReceptorEncoder(TestCase):

    def _construct_test_dataset(self, path, dataset_size: int = 50):
        receptors = [Receptor(chain_1=ReceptorSequence(sequence_aa="AAAA", locus='alpha'),
                              chain_2=ReceptorSequence(sequence_aa="ATA", locus='beta'),
                              metadata={"l1": 1}, receptor_id=str("1"), cell_id="1", chain_pair=ChainPair.TRA_TRB),
                     Receptor(chain_1=ReceptorSequence(sequence_aa="ATA", locus='alpha'),
                              chain_2=ReceptorSequence(sequence_aa="ATT", locus='beta'),
                              metadata={"l1": 2}, receptor_id="2", cell_id="2", chain_pair=ChainPair.TRA_TRB)]

        PathBuilder.build(path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        dataset = ReceptorDataset.build_from_objects(receptors, path)
        return dataset, lc

    def test(self):
        path = EnvironmentSettings.tmp_test_path / "onehot_sequence_1/"
        PathBuilder.remove_old_and_build(path)

        dataset, lc = self._construct_test_dataset(path)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False, 'sequence_type': 'amino_acid',
                                                         "distance_to_seq_middle": 6, "flatten": False,
                                                         'region_type': 'imgt_cdr3'})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True,
            model={},
        ))

        self.assertTrue(isinstance(encoded_data, ReceptorDataset))

        onehot_a = [1] + [0] * 19
        onehot_t = [0] * 16 + [1] + [0] * 3
        onehot_empty = [0] * 20

        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[0, 0]],
                             [onehot_a for i in range(4)])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[0, 1]],
                             [onehot_a, onehot_t, onehot_a, onehot_empty])

        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[1, 0]],
                             [onehot_a, onehot_t, onehot_a, onehot_empty])
        self.assertListEqual([list(item) for item in encoded_data.encoded_data.examples[1, 1]],
                             [onehot_a, onehot_t, onehot_t, onehot_empty])

        shutil.rmtree(path)

    def construct_test_flatten_dataset(self, path):
        receptors = [Receptor(chain_1=ReceptorSequence(sequence_aa="AAATTT", sequence_id="1a", locus='alpha'),
                              chain_2=ReceptorSequence(sequence_aa="ATATAT", sequence_id="1b",
                                                       locus='beta'),
                              metadata={"l1": 1},
                              cell_id="1", chain_pair=ChainPair.TRA_TRB),
                     Receptor(chain_1=ReceptorSequence(sequence_aa="AAAAAA", sequence_id="2a",
                                                       locus='alpha'),
                              chain_2=ReceptorSequence(sequence_aa="AAAAAA", sequence_id="2b",
                                                       locus='beta'),
                              metadata={"l1": 2},
                              cell_id="2", chain_pair=ChainPair.TRA_TRB),
                     Receptor(chain_1=ReceptorSequence(sequence_aa="AAAAAA", sequence_id="2a",
                                                       locus='alpha'),
                              chain_2=ReceptorSequence(sequence_aa="AAAAAA", sequence_id="2b",
                                                       locus='beta'),
                              metadata={"l1": 2},
                              cell_id="3", chain_pair=ChainPair.TRA_TRB)]

        return ReceptorDataset.build_from_objects(receptors, path)

    def test_receptor_flattened(self):
        path = EnvironmentSettings.tmp_test_path / "onehot_recep_flat/"

        PathBuilder.remove_old_and_build(path)

        dataset = self.construct_test_flatten_dataset(path)

        encoder = OneHotEncoder.build_object(dataset, **{"use_positional_info": False, "distance_to_seq_middle": None,
                                                         'sequence_type': 'amino_acid', "flatten": True,
                                                         'region_type': 'imgt_cdr3'})

        encoded_data = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=LabelConfiguration([Label(name="l1", values=[1, 0], positive_class="1")]),
            pool_size=1,
            learn_model=True,
            model={},
        ))

        self.assertTrue(isinstance(encoded_data, ReceptorDataset))

        onehot_a = [1.0] + [0.0] * 19
        onehot_t = [0.0] * 16 + [1.0] + [0] * 3

        self.assertListEqual(list(encoded_data.encoded_data.examples[0]),
                             onehot_a + onehot_a + onehot_a + onehot_t + onehot_t + onehot_t + onehot_a + onehot_t + onehot_a + onehot_t + onehot_a + onehot_t)
        self.assertListEqual(list(encoded_data.encoded_data.examples[1]), onehot_a * 12)
        self.assertListEqual(list(encoded_data.encoded_data.examples[2]), onehot_a * 12)

        self.assertListEqual(list(encoded_data.encoded_data.feature_names),
                             [f"{chain}_{pos}_{char}" for chain in ("alpha", "beta") for pos in range(6) for char in
                              EnvironmentSettings.get_sequence_alphabet()])

        shutil.rmtree(path)
