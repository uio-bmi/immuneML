import shutil
from unittest import TestCase

import numpy as np

from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.receptor_sequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.dsl.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.MatchedReferenceEncoder import MatchedReferenceEncoder
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.RepertoireBuilder import RepertoireBuilder


class TestMatchedReferenceEncoder(TestCase):
    def test__encode_new_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/matched_ref_encoder/"
        filenames, metadata = RepertoireBuilder.build([["AAAA", "AACA"], ["TTTA", "AAAA"]], path, {"default": np.array([1, 2])})
        dataset = Dataset(filenames=filenames)

        label_config = LabelConfiguration()
        label_config.add_label("default", [1, 2])

        encoded = MatchedReferenceEncoder._encode_new_dataset(dataset, EncoderParams(
            result_path=path,
            label_configuration=label_config,
            model={
                "reference_sequences": [ReceptorSequence("AAAA", metadata=SequenceMetadata())],
                "max_distance": 2,
                "summary": SequenceMatchingSummaryType.PERCENTAGE
            },
            filename="dataset.csv"
        ))

        self.assertTrue(all(all([val <= 1 for val in rep]) for rep in encoded.encoded_data.repertoires))
        self.assertEqual(2, encoded.encoded_data.repertoires.shape[0])

        shutil.rmtree(path)
