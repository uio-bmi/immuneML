import os
import shutil
from unittest import TestCase


from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestMotifEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _prepare_dataset(self, path):
        sequences = [ReceptorSequence(sequence_aa="AACC", sequence_id="1",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(sequence_aa="AGDD", sequence_id="2",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(sequence_aa="AAEE", sequence_id="3",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(sequence_aa="AGFF", sequence_id="4",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(sequence_aa="CCCC", sequence_id="5",
                                      metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(sequence_aa="DDDD", sequence_id="6",
                                      metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(sequence_aa="EEEE", sequence_id="7",
                                      metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(sequence_aa="FFFF", sequence_id="8",
                                      metadata=SequenceMetadata(custom_params={"l1": 2}))]


        PathBuilder.build(path)
        return SequenceDataset.build_from_objects(sequences, 100, PathBuilder.build(path / 'data'), 'd2')

    def test(self):
        path = EnvironmentSettings.tmp_test_path / "significant_motif_sequence_encoder_test/"
        dataset = self._prepare_dataset(path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2], positive_class=1)

        encoder = MotifEncoder.build_object(dataset, **{
                "min_positions": 1,
                "max_positions": 2,
                "min_precision": 0.9,
                "min_recall": 0.5,
                "min_true_positives": 1,
            })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoder_result/",
            label_config=lc,
            pool_size=4,
            learn_model=True,
            model={},
        ))

        self.assertEqual(8, encoded_dataset.encoded_data.examples.shape[0])
        self.assertTrue(all(identifier in encoded_dataset.encoded_data.example_ids
                            for identifier in ['1', '2', '3', '4', '5', '6', '7', '8']))

        self.assertListEqual(['0-A', '1-A', '1-G', '0&1-A&A', '0&1-A&G'], encoded_dataset.encoded_data.feature_names)

        self.assertListEqual([True, True, False, True, False], list(encoded_dataset.encoded_data.examples[0]))
        self.assertListEqual([True, False, True, False, True], list(encoded_dataset.encoded_data.examples[1]))
        self.assertListEqual([True, True, False, True, False], list(encoded_dataset.encoded_data.examples[2]))
        self.assertListEqual([True, False, True, False, True], list(encoded_dataset.encoded_data.examples[3]))
        self.assertListEqual([False, False, False, False, False], list(encoded_dataset.encoded_data.examples[4]))
        self.assertListEqual([False, False, False, False, False], list(encoded_dataset.encoded_data.examples[5]))
        self.assertListEqual([False, False, False, False, False], list(encoded_dataset.encoded_data.examples[6]))
        self.assertListEqual([False, False, False, False, False], list(encoded_dataset.encoded_data.examples[7]))

        shutil.rmtree(path)

    def _disabled_test_generalized(self):
        '''
        Old test, disabled as generalized_motifs option does not have a clear purpose as of now.
        '''

        path = EnvironmentSettings.tmp_test_path / "significant_motif_sequence_encoder_generalized/"
        dataset = self._prepare_dataset(path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2], positive_class=1)

        encoder = MotifEncoder.build_object(dataset, **{
                "min_positions": 1,
                "max_positions": 2,
                "min_precision": 0.9,
                "min_recall": 0.5,
                "generalized_motifs": True,
                "min_true_positives": 1,
            })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoder_result/",
            label_config=lc,
            pool_size=2,
            learn_model=True,
            model={},
        ))

        self.assertEqual(8, encoded_dataset.encoded_data.examples.shape[0])
        self.assertTrue(all(identifier in encoded_dataset.encoded_data.example_ids
                            for identifier in ['1', '2', '3', '4', '5', '6', '7', '8']))

        self.assertListEqual(['0-A', '1-A', '1-G', '0&1-A&A', '0&1-A&G', '0&1-A&AG'], encoded_dataset.encoded_data.feature_names)

        self.assertListEqual([True, True, False, True, False, True], list(encoded_dataset.encoded_data.examples[0]))
        self.assertListEqual([True, False, True, False, True, True], list(encoded_dataset.encoded_data.examples[1]))
        self.assertListEqual([True, True, False, True, False, True], list(encoded_dataset.encoded_data.examples[2]))
        self.assertListEqual([True, False, True, False, True, True], list(encoded_dataset.encoded_data.examples[3]))
        self.assertListEqual([False, False, False, False, False, False], list(encoded_dataset.encoded_data.examples[4]))
        self.assertListEqual([False, False, False, False, False, False], list(encoded_dataset.encoded_data.examples[5]))
        self.assertListEqual([False, False, False, False, False, False], list(encoded_dataset.encoded_data.examples[6]))
        self.assertListEqual([False, False, False, False, False, False], list(encoded_dataset.encoded_data.examples[7]))

        shutil.rmtree(path)