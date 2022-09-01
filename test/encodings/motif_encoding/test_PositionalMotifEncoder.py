import os
import shutil
from unittest import TestCase


from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.PositionalMotifEncoder import PositionalMotifEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestPositionalMotifEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _prepare_dataset(self, path):
        sequences = [ReceptorSequence(amino_acid_sequence="AACC", identifier="1",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="AADD", identifier="2",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="AAEE", identifier="3",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="AAFF", identifier="4",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="CCCC", identifier="5",
                                      metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="DDDD", identifier="6",
                                      metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="EEEE", identifier="7",
                                      metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="FFFF", identifier="8",
                                      metadata=SequenceMetadata(custom_params={"l1": 2}))]


        PathBuilder.build(path)
        return SequenceDataset.build_from_objects(sequences, 100, PathBuilder.build(path / 'data'), 'd2')

    def test(self):
        path = EnvironmentSettings.tmp_test_path / "positional_motif_sequence_encoder/test/"
        dataset = self._prepare_dataset(path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2], positive_class=1)

        encoder = PositionalMotifEncoder.build_object(dataset, **{
                "max_positions": 3,
                "min_precision": 0.9,
                "min_recall": 0.5,
                "min_true_positives": 1,
                "generalize_motifs": True,
            })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoder_result/",
            label_config=lc,
            pool_size=2,
            learn_model=True,
            model={},
            filename="dataset.csv"
        ))

        self.assertEqual(8, encoded_dataset.encoded_data.examples.shape[0])
        self.assertTrue(all(identifier in encoded_dataset.encoded_data.example_ids
                            for identifier in ['1', '2', '3', '4', '5', '6', '7', '8']))

        self.assertListEqual(['0-A', '1-A', '0&1-A&A'], encoded_dataset.encoded_data.feature_names)

        for i in range(4):
            self.assertListEqual([True] * 3, list(encoded_dataset.encoded_data.examples[i]))

        for i in range(5, 8):
            self.assertListEqual([False] * 3, list(encoded_dataset.encoded_data.examples[i]))

        shutil.rmtree(path)
