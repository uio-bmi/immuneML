import os
import shutil
from unittest import TestCase


from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.SimilarToPositiveSequenceEncoder import SimilarToPositiveSequenceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestSimilarToPositiveSequenceEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _prepare_dataset(self, path):
        sequences = [ReceptorSequence(amino_acid_sequence="AACC", identifier="1",
                                      metadata=SequenceMetadata(custom_params={"l1": "yes"})),
                     ReceptorSequence(amino_acid_sequence="AGDD", identifier="2",
                                      metadata=SequenceMetadata(custom_params={"l1": "yes"})),
                     ReceptorSequence(amino_acid_sequence="AAEE", identifier="3",
                                      metadata=SequenceMetadata(custom_params={"l1": "yes"})),
                     ReceptorSequence(amino_acid_sequence="CCCC", identifier="4",
                                      metadata=SequenceMetadata(custom_params={"l1": "no"})),
                     ReceptorSequence(amino_acid_sequence="AGDE", identifier="5",
                                      metadata=SequenceMetadata(custom_params={"l1": "no"})),
                     ReceptorSequence(amino_acid_sequence="EEEE", identifier="6",
                                      metadata=SequenceMetadata(custom_params={"l1": "no"}))]


        PathBuilder.build(path)
        return SequenceDataset.build_from_objects(sequences, 100, PathBuilder.build(path / 'data'), 'd2')

    def test(self):
        path = EnvironmentSettings.tmp_test_path / "significant_motif_sequence_encoder_test/"
        dataset = self._prepare_dataset(path)

        lc = LabelConfiguration()
        lc.add_label("l1", ["yes", "no"], positive_class="yes")

        encoder = SimilarToPositiveSequenceEncoder.build_object(dataset, **{"hamming_distance": 1})

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoder_result/",
            label_config=lc,
            pool_size=4,
            learn_model=True,
            model={},
        ))

        self.assertEqual(6, encoded_dataset.encoded_data.examples.shape[0])
        self.assertTrue(all(identifier in encoded_dataset.encoded_data.example_ids
                            for identifier in ['1', '2', '3', '4', '5', '6']))

        self.assertListEqual(["similar_to_positive_sequence"], encoded_dataset.encoded_data.feature_names)

        self.assertListEqual([True, True, True, False, True, False], list(encoded_dataset.encoded_data.examples))

        shutil.rmtree(path)
