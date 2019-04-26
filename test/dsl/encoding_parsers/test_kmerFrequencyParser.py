from unittest import TestCase

from source.dsl.encoding_parsers.KmerFrequencyParser import KmerFrequencyParser
from source.encodings.kmer_frequency.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType


class TestKmerFrequencyParser(TestCase):
    def test_parse(self):
        params = {
            "normalization_type": "l2",
            "reads": "unique",
            "sequence_encoding_type": "identity",
            "k": 3
        }

        parsed = KmerFrequencyParser.parse(params)
        self.assertEqual(NormalizationType.L2, parsed["normalization_type"])
        self.assertEqual(3, parsed["k"])
        self.assertEqual(SequenceEncodingType.IDENTITY, parsed["sequence_encoding_type"])
