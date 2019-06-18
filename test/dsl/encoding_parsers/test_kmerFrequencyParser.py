from unittest import TestCase

from source.dsl.encoding_parsers.KmerFrequencyParser import KmerFrequencyParser
from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType


class TestKmerFrequencyParser(TestCase):
    def test_parse(self):
        params = {
            "k": 3
        }

        parsed, specs = KmerFrequencyParser.parse(params)
        self.assertEqual(NormalizationType.L2, parsed["normalization_type"])
        self.assertEqual(3, parsed["k"])
        self.assertEqual(SequenceEncodingType.CONTINUOUS_KMER, parsed["sequence_encoding"])
