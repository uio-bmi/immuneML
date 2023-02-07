import os
import shutil
from pathlib import Path
from unittest import TestCase


from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
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
        sequences = [ReceptorSequence(amino_acid_sequence="AACC", identifier="5",
                                      metadata=SequenceMetadata(custom_params={"l1": "yes"})),
                     ReceptorSequence(amino_acid_sequence="AGDD", identifier="3",
                                      metadata=SequenceMetadata(custom_params={"l1": "yes"})),
                     ReceptorSequence(amino_acid_sequence="AAEE", identifier="4",
                                      metadata=SequenceMetadata(custom_params={"l1": "yes"})),
                     ReceptorSequence(amino_acid_sequence="CCCC", identifier="1",
                                      metadata=SequenceMetadata(custom_params={"l1": "no"})),
                     ReceptorSequence(amino_acid_sequence="AGDE", identifier="2",
                                      metadata=SequenceMetadata(custom_params={"l1": "no"})),
                     ReceptorSequence(amino_acid_sequence="EEEE", identifier="6",
                                      metadata=SequenceMetadata(custom_params={"l1": "no"}))]


        PathBuilder.build(path)
        dataset = SequenceDataset.build_from_objects(sequences, 100, PathBuilder.build(path / 'data'), 'd2')

        lc = LabelConfiguration()
        lc.add_label("l1", ["yes", "no"], positive_class="yes")

        return dataset, lc

    def _get_encoder_params(self, path, lc):
        return  EncoderParams(
            result_path=path / "encoder_result/",
            label_config=lc,
            pool_size=4,
            learn_model=True,
            model={},
        )

    def test_generate(self, compairr_path=None):
        path_suffix = "compairr" if compairr_path else "no_compairr"
        path = EnvironmentSettings.tmp_test_path / f"significant_motif_sequence_encoder_test_{path_suffix}/"
        dataset, lc = self._prepare_dataset(path)

        default_params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "encodings/", "similar_to_positive_sequence")

        encoder = SimilarToPositiveSequenceEncoder.build_object(dataset, **{**default_params, **{"hamming_distance": 1,
                                                                                                 "compairr_path": compairr_path,
                                                                                                 "ignore_genes": True}})

        encoded_dataset = encoder.encode(dataset, self._get_encoder_params(path, lc))

        self.assertEqual(6, encoded_dataset.encoded_data.examples.shape[0])
        self.assertTrue(all(identifier in encoded_dataset.encoded_data.example_ids
                            for identifier in ["1", "2", "3", "4", "5", "6"]))

        self.assertListEqual(["similar_to_positive_sequence"], encoded_dataset.encoded_data.feature_names)

        self.assertListEqual([True, True, True, False, True, False], list(encoded_dataset.encoded_data.examples))

        shutil.rmtree(path)

    def test_generate_with_compairr(self):
        compairr_paths = [Path("/usr/local/bin/compairr"), Path("./compairr/src/compairr")]

        for compairr_path in compairr_paths:
            if compairr_path.exists():
                self.test_generate(str(compairr_path))
