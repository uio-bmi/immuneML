from unittest import TestCase

from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.dsl.encoding_parsers.EncodingParser import EncodingParser
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder


class TestEncodingParser(TestCase):
    def test_parse_encoder(self):
        param = {
            "encodings": {
                "KF": {
                    "type": "KmerFrequency",
                    "params": {
                        "normalization_type": "relative_frequency",
                        "sequence_encoding": "identity",
                        "k": 3
                    },
                    "dataset": "d1"
                }
            }
        }

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, {})
        symbol_table, specs = EncodingParser.parse(param, symbol_table)
        self.assertTrue(isinstance(symbol_table.get("KF"), dict))
        self.assertEqual(4, len(symbol_table.get("KF").keys()))
        self.assertTrue(isinstance(symbol_table.get("KF")["params"], dict))
        self.assertEqual(KmerFrequencyEncoder, symbol_table.get("KF")["encoder"])

        self.assertTrue("reads" in specs["KF"]["params"].keys())
        self.assertEqual("unique", specs["KF"]["params"]["reads"])
        self.assertEqual("KmerFrequency", specs["KF"]["type"])
