from unittest import TestCase

from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.dsl.ParameterParser import ParameterParser
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType


class TestParameterParser(TestCase):
    def test_parse(self):
        params = {"a": 3}

        parsed_params, specs = ParameterParser.parse(params)
        self.assertTrue("a" in parsed_params.keys())
        self.assertEqual(1, len(parsed_params.keys()))

        params = {
            "normalization_type": "l2",
            "reads": "unique",
            "sequence_encoding": "continuous_kmer",
            "k": 3,
            "k_left": 1,
            "k_right": 1,
            "max_gap": 0,
            "min_gap": 0,
            "batch_size": 1
        }

        parsed_params, specs = ParameterParser.parse(params, "KmerFrequency", "encoding_parsers/")

        self.assertEqual(3, parsed_params["k"])
        self.assertEqual(0, parsed_params["min_gap"])
        self.assertEqual(NormalizationType.L2, parsed_params["normalization_type"])
        self.assertEqual(SequenceEncodingType.CONTINUOUS_KMER, parsed_params["sequence_encoding"])
        self.assertEqual("continuous_kmer", specs["sequence_encoding"])
