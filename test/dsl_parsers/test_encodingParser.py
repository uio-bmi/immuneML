from unittest import TestCase

from source.dsl_parsers.EncodingParser import EncodingParser
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder


class TestEncodingParser(TestCase):
    def test_parse_encoder(self):
        param = {
            "encoder": "KmerFrequencyEncoder",
            "encoder_params": {
                "normalization_type": "relative_frequency",
                "reads": "unique",
                "sequence_encoding": "identity",
                "k": 3
            }
        }

        encoder, encoder_params = EncodingParser.parse_encoder(param)
        self.assertTrue(isinstance(encoder, KmerFrequencyEncoder))
        self.assertTrue(all([key in encoder_params.keys() for key in ["k", "sequence_encoding", "normalization_type"]]))
