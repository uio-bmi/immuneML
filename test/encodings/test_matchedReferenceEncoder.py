import shutil
from unittest import TestCase

import numpy as np

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.MatchedReferenceEncoder import MatchedReferenceEncoder
from source.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.RepertoireBuilder import RepertoireBuilder


class TestMatchedReferenceEncoder(TestCase):
    def test__encode_new_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/matched_ref_encoder/"
        repertoires, metadata = RepertoireBuilder.build([["AAAA", "AACA"], ["TTTA", "AAAA"]], path, {"default": np.array([1, 2])})
        dataset = RepertoireDataset(repertoires=repertoires)

        label_config = LabelConfiguration()
        label_config.add_label("default", [1, 2])

        encoder = MatchedReferenceEncoder.create_encoder(dataset, {
                "reference_sequences": [ReceptorSequence("AAAA",
                                                         metadata=SequenceMetadata(chain=Chain.A.value, v_gene="V12", j_gene="J1"))],
                "max_edit_distance": 2,
                "summary": SequenceMatchingSummaryType.PERCENTAGE
            })

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_configuration=label_config,
            model={},
            filename="dataset.csv"
        ))

        self.assertTrue(all(all([val <= 1 for val in rep]) for rep in encoded.encoded_data.examples))
        self.assertEqual(2, encoded.encoded_data.examples.shape[0])
        self.assertTrue(isinstance(encoder, MatchedReferenceEncoder))

        shutil.rmtree(path)
